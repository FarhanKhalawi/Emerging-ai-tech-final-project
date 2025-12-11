import os
import time
import random
import math
from typing import Any, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

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

from config import Config


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def format_example(example: Dict[str, Any]) -> Dict[str, str]:

    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {"prompt": prompt, "label": output_text}


def tokenize_function(batch: Dict[str, List[str]], tokenizer, max_length: int) -> Dict[str, Any]:
    

    texts = [p + " " + l for p, l in zip(batch["prompt"], batch["label"])]
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,  
    )


def main():
    # -------------------------------
    # Checks and seeding
    # -------------------------------
    if Config.HF_TOKEN is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in the environment.")

    set_all_seeds(Config.SEED)

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    # -------------------------------
    # Load dataset and split
    # -------------------------------
    dataset = load_dataset(Config.DATASET_NAME)
    full_train = dataset["train"].shuffle(seed=Config.SEED)

    train_end = Config.TRAIN_SAMPLES
    val_end   = train_end + Config.VAL_SAMPLES
    test_end  = val_end + Config.TEST_SAMPLES

    train_data = full_train.select(range(0, train_end))
    val_data   = full_train.select(range(train_end, val_end))
    test_data  = full_train.select(range(val_end, test_end))

    print("Train size:", len(train_data))
    print("Validation size:", len(val_data))
    print("Test size:", len(test_data))

    # -------------------------------
    # Format data: prompt + label
    # -------------------------------
    train_data = train_data.map(format_example)
    val_data   = val_data.map(format_example)
    test_data  = test_data.map(format_example)

    print("Example formatted prompt:")
    print(train_data[0]["prompt"])
    print("-----------------------------")
    print("Label:")
    print(train_data[0]["label"])

    # -------------------------------
    # Load tokenizer
    # -------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        token=Config.HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------------
    # Tokenize datasets
    # -------------------------------
    print("Tokenizing train...")
    train_tokenized = train_data.map(
        lambda batch: tokenize_function(batch, tokenizer, Config.MAX_LENGTH),
        batched=True,
        remove_columns=train_data.column_names,
    )

    print("Tokenizing validation...")
    val_tokenized = val_data.map(
        lambda batch: tokenize_function(batch, tokenizer, Config.MAX_LENGTH),
        batched=True,
        remove_columns=val_data.column_names,
    )

    print("Tokenizing test...")
    test_tokenized = test_data.map(
        lambda batch: tokenize_function(batch, tokenizer, Config.MAX_LENGTH),
        batched=True,
        remove_columns=test_data.column_names,
    )

    print("Example tokenized sample (first 10 token ids):")
    print({k: v[:10] for k, v in train_tokenized[0].items()})

    # -------------------------------
    # Data collator 
    # -------------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -------------------------------
    # Load base model + apply LoRA
    # -------------------------------
    print("Loading model:", Config.MODEL_NAME)
    dtype = torch.float16 if (Config.USE_FP16 and torch.cuda.is_available()) else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        token=Config.HF_TOKEN,
        dtype=dtype,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        task_type=Config.LORA_TASK_TYPE,
    )
    model = get_peft_model(base_model, lora_config)
    print("LoRA model ready.")

    # -------------------------------
    # TrainingArguments
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=Config.CHECKPOINT_DIR,
        per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=Config.WARMUP_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        logging_steps=Config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        save_total_limit=Config.SAVE_TOTAL_LIMIT,
        seed=Config.SEED,
        fp16=True if (Config.USE_FP16 and torch.cuda.is_available()) else False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,       
    )

    # -------------------------------
    # Train, time, and GPU usage
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
    # Evaluate on validation set
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
    # 11. Plot training & validation loss
    # -------------------------------
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for entry in trainer.state.log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_losses.append(entry["eval_loss"])

    plt.figure()
    if train_losses:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(Config.PLOTS_DIR, "loss_curves.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {plot_path}")

    # -------------------------------
    # Save final (best) model + tokenizer
    # -------------------------------
    trainer.save_model(Config.BEST_MODEL_DIR)
    tokenizer.save_pretrained(Config.CHECKPOINT_DIR)
    print(f"Training finished. Best model saved to {Config.BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()
