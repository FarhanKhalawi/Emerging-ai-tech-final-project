import os
import random
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

SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# 0. Hugging Face token
# -------------------------------
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

if HF_TOKEN is None:
    raise ValueError(
        "HUGGINGFACE_HUB_TOKEN is not set. "
        "Run: export HUGGINGFACE_HUB_TOKEN=hf_your_token_here"
    )


def main():
    set_seed(SEED)

    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    dataset = load_dataset("yahma/alpaca-cleaned")
    print(dataset)

    # -------------------------------
    # 2. Shuffle and split dataset
    # -------------------------------
    full_train = dataset["train"].shuffle(seed=SEED)

    train_data = full_train.select(range(0, 10000))
    val_data   = full_train.select(range(10000, 12000))
    test_data  = full_train.select(range(12000, 14000))

    print("Train size:", len(train_data))
    print("Validation size:", len(val_data))
    print("Test size:", len(test_data))

    # -------------------------------
    # 3. Format examples into a single text (prompt + output)
    # -------------------------------
    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        instruction = example["instruction"]
        input_text  = example["input"]
        output_text = example["output"]

        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"

        # full sequence that model will see: prompt + answer
        full_text = prompt + " " + output_text
        return {"text": full_text}

    train_data = train_data.map(format_example)
    val_data   = val_data.map(format_example)
    test_data  = test_data.map(format_example)

    print("Example formatted text:")
    print(train_data[0]["text"])

    # -------------------------------
    # 4. Load tokenizer
    # -------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=HF_TOKEN,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------
    # 5. Tokenize datasets (NO manual labels)
    # -------------------------------
    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=512,
            truncation=True,
            padding=False,   # padding is done later in the collator
        )

    print("Tokenizing train...")
    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=train_data.column_names,  # removes "text"
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
    #    - Creates labels from input_ids internally
    # -------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # causal LM (next-token prediction), not masked LM
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

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,              # rank (you analyse this in the report)
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

    print("Starting training...")
    trainer.train()

    # -------------------------------
    # 9. Save final model (LoRA adapter)
    # -------------------------------
    out_dir = "./outputs/lora_llama"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Training finished and model saved to: {out_dir}")


if __name__ == "__main__":
    main()
