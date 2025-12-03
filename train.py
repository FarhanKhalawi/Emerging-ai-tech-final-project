import os
import random
import time
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch
import math
import matplotlib.pyplot as plt

SEED = 42

# -------------------------------
# 0. Read Hugging Face token
# -------------------------------
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HUGGINGFACE_HUB_TOKEN is not set. "
        "Run: export HUGGINGFACE_HUB_TOKEN=hf_your_token_here"
    )

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = load_dataset("yahma/alpaca-cleaned")
print(dataset)

# -------------------------------
# 2. Shuffle and split dataset (10k / 2k / 2k)
# -------------------------------
random.seed(SEED)
full_train = dataset["train"].shuffle(seed=SEED)

train_data = full_train.select(range(0, 10000))
val_data   = full_train.select(range(10000, 12000))
test_data  = full_train.select(range(12000, 14000))

print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))

# -------------------------------
# 3. Format examples into prompt + label (for inspection)
#    Training text will be prompt + label.
# -------------------------------
def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {
        "prompt": prompt,
        "label": output_text,
    }

train_data = train_data.map(format_example)
val_data   = val_data.map(format_example)
test_data  = test_data.map(format_example)

print("Example formatted prompt:")
print(train_data[0]["prompt"])
print("-----")
print("Label:")
print(train_data[0]["label"])

# -------------------------------
# 4. Load tokenizer (with token)
# -------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token=HF_TOKEN,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# 5. Tokenize datasets
#    Combine prompt + label into one training sequence.
# -------------------------------
def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
    texts = [p + " " + l for p, l in zip(batch["prompt"], batch["label"])]
    return tokenizer(
        texts,
        max_length=512,
        truncation=True,
        padding=False,  # dynamic padding in collator
    )

print("Tokenizing train...")
train_tokenized = train_data.map(
    tokenize_function,
    batched=True,
    remove_columns=train_data.column_names,
)

print("Tokenizing validation...")
val_tokenized = val_data.map(
    tokenize_function,
    batched=True,
    remove_columns=val_data.column_names,
)

print("Tokenizing test...")
test_tokenized = test_data.map(
    tokenize_function,
    batched=True,
    remove_columns=test_data.column_names,
)

print("Example tokenized sample:")
print({k: v[:10] for k, v in train_tokenized[0].items()})

# -------------------------------
# 6. Data collator for causal LM
#    (creates labels from input_ids internally)
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------------------------------
# 7. Load model with LoRA 
# -------------------------------
print("Loading Llama-3.2-1B model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=8,              # rank (to analyse in report)
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print("LoRA model ready.")

# -------------------------------
# 8. Training setup
# -------------------------------
training_args = TrainingArguments(
    output_dir="./outputs/checkpoints",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=1,  # change to 3 for full run if you want
    logging_steps=50,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -------------------------------
# 8.1 Train, track time & GPU usage
# -------------------------------
start_time = time.time()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

print("Starting training...")
trainer.train()
end_time = time.time()
total_time_sec = end_time - start_time
print(f"Total training time: {total_time_sec:.2f} seconds")

if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    print(f"Peak GPU memory usage: {peak_mem_gb:.2f} GB")

# -------------------------------
# 8.2 Evaluate on validation set (for reporting)
# -------------------------------
eval_metrics = trainer.evaluate(eval_dataset=val_tokenized)
print("Validation metrics:", eval_metrics)

if "eval_loss" in eval_metrics:
    try:
        ppl = math.exp(eval_metrics["eval_loss"])
        print(f"Validation perplexity: {ppl:.4f}")
    except OverflowError:
        print("Validation perplexity: overflow (loss too large)")

# -------------------------------
# 8.3 Plot training & validation loss
# -------------------------------
os.makedirs("./outputs/plots", exist_ok=True)

train_steps: List[int] = []
train_losses: List[float] = []
eval_steps: List[int] = []
eval_losses: List[float] = []

for entry in trainer.state.log_history:
    # Training loss entries
    if "loss" in entry:
        step = entry.get("step", len(train_steps))
        train_steps.append(step)
        train_losses.append(entry["loss"])

    # Evaluation loss entries (from evaluate())
    if "eval_loss" in entry:
        if "step" in entry:
            step = entry["step"]
        elif train_steps:
            # If no step logged, place eval at last training step
            step = train_steps[-1]
        else:
            step = len(eval_steps)
        eval_steps.append(step)
        eval_losses.append(entry["eval_loss"])

plt.figure()
if train_losses:
    plt.plot(train_steps, train_losses, label="train_loss")
if eval_losses:
    # marker + dashed line so it's clearly visible
    plt.plot(eval_steps, eval_losses, marker="o", linestyle="--", label="eval_loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plot_path = "./outputs/plots/loss_curves.png"
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"Saved loss curves to {plot_path}")

# -------------------------------
# 9. Save final model checkpoint
# -------------------------------
os.makedirs("./outputs", exist_ok=True)
trainer.save_model("./outputs/best_model.pt")
tokenizer.save_pretrained("./outputs/checkpoints")
print("Training finished and model saved to ./outputs/best_model.pt")
