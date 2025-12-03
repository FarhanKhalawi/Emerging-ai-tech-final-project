import os
import random
import math
import json
from typing import Any, Dict, List
from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
import torch

SEED = 42
MODEL_DIR = "./outputs/best_model.pt"   # LoRA adapter saved by train.py

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set.")


def build_prompt_and_label(example: Dict[str, Any]) -> Dict[str, str]:
    instruction = example["instruction"]
    input_text  = example["input"]
    output_text = example["output"]

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {"prompt": prompt, "label": output_text}


def word_f1(pred: str, gold: str) -> float:
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 1. Dataset and test split
    dataset = load_dataset("yahma/alpaca-cleaned")
    full_train = dataset["train"].shuffle(seed=SEED)
    test_data  = full_train.select(range(12000, 14000))
    print("Test size:", len(test_data))

    # 2. Build test texts (prompt + label) for perplexity
    def to_text(batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        texts = []
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            if inp:
                prompt = f"Instruction: {inst}\nInput: {inp}\nResponse:"
            else:
                prompt = f"Instruction: {inst}\nResponse:"
            texts.append(prompt + " " + out)
        return {"text": texts}

    test_text_ds = test_data.map(
        to_text,
        batched=True,
        remove_columns=test_data.column_names,
    )

    # 3. Tokenizer + models (base + fine-tuned)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        token=HF_TOKEN,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    ft_model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    ft_model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Tokenize test set
    def tokenize_text(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=512,
            truncation=True,
            padding=False,
        )

    test_tokenized = test_text_ds.map(
        tokenize_text,
        batched=True,
        remove_columns=test_text_ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Perplexity for base and fine-tuned
    def compute_ppl(model, name: str) -> Dict[str, Any]:
        args = TrainingArguments(
            output_dir=f"./outputs/eval_{name}",
            per_device_eval_batch_size=2,
            seed=SEED,
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_tokenized,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        metrics = trainer.evaluate()
        ppl = None
        if "eval_loss" in metrics:
            try:
                ppl = math.exp(metrics["eval_loss"])
            except OverflowError:
                ppl = None
        return {"metrics": metrics, "perplexity": ppl}

    print("Computing perplexity for BASE model...")
    base_results = compute_ppl(base_model, "base")

    print("Computing perplexity for FINE-TUNED (LoRA) model...")
    ft_results = compute_ppl(ft_model, "lora_ft")

    # 6. F1 on subset of test examples (10 examples)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model.to(device)
    ft_model.eval()

    subset = test_data.select(range(10))
    examples_out = []

    for i, ex in enumerate(subset):
        pl = build_prompt_and_label(ex)
        prompt = pl["prompt"]
        gold = pl["label"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids = ft_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        if full_text.startswith(prompt):
            pred_answer = full_text[len(prompt):].strip()
        else:
            pred_answer = full_text

        f1 = word_f1(pred_answer, gold)
        examples_out.append(
            {
                "index": i,
                "instruction": ex["instruction"],
                "input": ex["input"],
                "reference_output": gold,
                "model": "llama-3.2-1B-lora",
                "strategy": "greedy",
                "model_output": pred_answer,
                "f1": f1,
            }
        )
        print(f"[{i}] F1 = {f1:.4f}")

    avg_f1 = sum(e["f1"] for e in examples_out) / len(examples_out)
    print(f"Average F1 over 10 examples: {avg_f1:.4f}")

    # 7. Save JSON: outputs/generations/test_set_evaluation.json
    os.makedirs("./outputs/generations", exist_ok=True)
    out = {
        "config": {
            "seed": SEED,
            "base_model": "meta-llama/Llama-3.2-1B",
            "adapter_path": MODEL_DIR,
        },
        "metrics": {
            "base": base_results,
            "fine_tuned": {
                "metrics": ft_results["metrics"],
                "perplexity": ft_results["perplexity"],
                "avg_f1": avg_f1,
            },
        },
        "examples": examples_out,
    }

    out_path = "./outputs/generations/test_set_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved test set evaluation to {out_path}")


if __name__ == "__main__":
    main()
