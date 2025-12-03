import os
import time
import math
import random
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 0. Global config / hyperparameters  (document these in report)
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "yahma/alpaca-cleaned"
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

MAX_LENGTH = 512
TRAIN_SIZE = 10_000
VAL_SIZE = 2_000
TEST_SIZE = 2_000

SEED = 42
LR = 2e-4
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

# LoRA hyperparameters (rank is important for the report)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Read HF token from environment
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN is None:
    raise ValueError(
        "HUGGINGFACE_HUB_TOKEN is not set. "
        "Run: export HUGGINGFACE_HUB_TOKEN=hf_your_token_here"
    )


def set_global_seed(seed: int) -> None:
    """Ensure reproducibility across Python, NumPy, PyTorch, and HF Trainer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # transformers helper


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert Alpaca (instruction, input, output) into a single text prompt + label.

    Training will be standard causal LM on [PROMPT + RESPONSE].
    """
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {"prompt": prompt, "label": output_text}


def tokenize_function(batch: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenize [prompt + label] without padding; dynamic padding is done by the collator.
    """
    texts = [p + " " + l for p, l in zip(batch["prompt"], batch["label"])]
    return tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,  # dynamic padding in data collator
    )


def count_trainable_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # -------------------------------
    # 1. Reproducibility
    # -------------------------------
    set_global_seed(SEED)

    # -------------------------------
    # 2. Load and split dataset (5:1:1)
    # -------------------------------
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    full_train = dataset["train"].shuffle(seed=SEED)

    train_data = full_train.select(range(0, TRAIN_SIZE))
    val_data = full_train.select(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))
    test_data = full_train.select(range(TRAIN_SIZE + VAL_SIZE,
                                        TRAIN_SIZE + VAL_SIZE + TEST_SIZE))

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

    # -------------------------------
    # 3. Format examples for instruction tuning
    # -------------------------------
    train_data = train_data.map(format_example)
    val_data = val_data.map(format_example)
    test_data = test_data.map(format_example)

    print("Example formatted prompt:")
    print(train_data[0]["prompt"])
    print("-----")
    print("Label:")
    print(train_data[0]["label"])

    # -------------------------------
    # 4. Load tokenizer
    # -------------------------------
    print(f"Loading tokenizer for {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

    # Ensure a pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------
    # 5. Tokenize datasets
    # -------------------------------
    print("Tokenizing train set...")
    train_tokenized = train_data.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=train_data.column_names,
    )

    print("Tokenizing validation set...")
    val_tokenized = val_data.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=val_data.column_names,
    )

    print("Tokenizing test set...")
    test_tokenized = test_data.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
        remove_columns=test_data.column_names,
    )

    print("Example tokenized sample:")
    print({k: v[:10] for k, v in train_tokenized[0].items()})

    # -------------------------------
    # 6. Data collator for causal LM
    # -------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )

    # -------------------------------
    # 7. Load base model and apply LoRA
    # -------------------------------
    print(f"Loading base model: {MODEL_NAME} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_info = count_trainable_parameters(model)
    print(
        f"Total params: {trainable_info['total']:,} | "
        f"Trainable (LoRA): {trainable_info['trainable']:,}"
    )

    # -------------------------------
    # 8. Training setup
    #    - 3 epochs
    #    - track validation loss
    #    - save *best* checkpoint by eval_loss
    # -------------------------------
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LR,
        logging_dir=LOG_DIR,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keep best + last
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
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
    # 9. Training with time & GPU stats
    # -------------------------------
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Starting training...")
    train_result = trainer.train()
    end_time = time.time()
    total_time_sec = end_time - start_time

    print(f"Total training time: {total_time_sec:.2f} seconds")

    peak_mem_gb = None
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        print(f"Peak GPU memory usage: {peak_mem_gb:.2f} GB")

    # -------------------------------
    # 10. Evaluate best model on validation set
    # -------------------------------
    eval_metrics = trainer.evaluate(eval_dataset=val_tokenized)
    print("Validation metrics:", eval_metrics)

    val_ppl = None
    if "eval_loss" in eval_metrics:
        try:
            val_ppl = math.exp(eval_metrics["eval_loss"])
            print(f"Validation perplexity: {val_ppl:.4f}")
        except OverflowError:
            print("Validation perplexity: overflow (loss too large)")

    # -------------------------------
    # 11. Save best model + adapter + training stats
    # -------------------------------
    # At this point, load_best_model_at_end=True ensures trainer.model
    # contains the best checkpoint (lowest eval_loss).
    trainer.save_model(BEST_MODEL_DIR)      # full HF checkpoint (config + adapter weights)
    tokenizer.save_pretrained(BEST_MODEL_DIR)

    # Optional: save raw state_dict as single .pt file (matches example tree)
    torch.save(trainer.model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

    # Adapter-only weights (for quick loading with base model)
    trainer.model.save_pretrained(ADAPTER_DIR)

    # Save training statistics for your report (LoRA rank analysis)
    train_stats = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LR,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "trainable_parameters": trainable_info["trainable"],
        "total_parameters": trainable_info["total"],
        "total_training_time_sec": float(total_time_sec),
        "peak_gpu_memory_gb": float(peak_mem_gb) if peak_mem_gb is not None else None,
        "final_eval_metrics": {k: float(v) for k, v in eval_metrics.items()},
        "validation_perplexity": float(val_ppl) if val_ppl is not None else None,
    }

    import json

    with open(os.path.join(OUTPUT_DIR, "train_stats.json"), "w") as f:
        json.dump(train_stats, f, indent=2)

    print("Training finished. Best model and stats saved in ./outputs/")


if __name__ == "__main__":
    main()
