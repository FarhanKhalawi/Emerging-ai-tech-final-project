import os
import random
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import torch

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
# 2. Shuffle and split dataset
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
# 3. Format examples into prompt + label
# -------------------------------
def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"]
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

# -------------------------------
# 5. Tokenize datasets
# -------------------------------
def tokenize_function(example):
    model_inputs = tokenizer(
        example["prompt"],
        max_length=512,
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["label"],
            max_length=512,
            truncation=True,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
print({k: v[:10] if isinstance(v, list) else v for k, v in train_tokenized[0].items()})

# -------------------------------
# 6. Prepare for PyTorch
# -------------------------------
train_tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
val_tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# -------------------------------
# 7. Load model with LoRA (using token)
# -------------------------------
print("Loading Llama-3.2-1B model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    token=HF_TOKEN,
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA configuration
lora_config = LoraConfig(
    r=8,              # you will analyse this rank in the report
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
    num_train_epochs=1,      # later change to 3 for the real run
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
)

print("Starting training...")
trainer.train()

# -------------------------------
# 9. Save final model
# -------------------------------
trainer.save_model("./outputs/best_model.pt")
tokenizer.save_pretrained("./outputs/checkpoints")
print("Training finished and model saved.")
