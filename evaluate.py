import os
import random
import math
import json
import statistics
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
from transformers.utils.logging import set_verbosity_error
from peft import PeftModel
import torch

from config import Config


# -------------------------------
#  Helper: build prompt + label
# -------------------------------
def build_prompt_and_label(example: Dict[str, Any]) -> Dict[str, str]:
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    if input_text:
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    else:
        prompt = f"Instruction: {instruction}\nResponse:"

    return {"prompt": prompt, "label": output_text}


# -------------------------------
#  Helper: token-level F1
# -------------------------------
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
    # -------------------------------
    # Checks, seeding and logging level
    # -------------------------------
    if Config.HF_TOKEN is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in environment.")

    set_verbosity_error()

    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    # -------------------------------
    # Load dataset and TEST split
    # -------------------------------
    dataset = load_dataset(Config.DATASET_NAME)
    full_train = dataset["train"].shuffle(seed=Config.SEED)

    train_end = Config.TRAIN_SAMPLES
    val_end = train_end + Config.VAL_SAMPLES
    test_end = val_end + Config.TEST_SAMPLES

    test_data = full_train.select(range(val_end, test_end))
    print("Test size:", len(test_data))

    # -------------------------------
    # Build test texts for perplexity
    # -------------------------------
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

    # -------------------------------
    # Tokenizer + models 
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        token=Config.HF_TOKEN,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        token=Config.HF_TOKEN,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    ft_model = PeftModel.from_pretrained(base_model, Config.BEST_MODEL_DIR)
    ft_model.config.pad_token_id = tokenizer.pad_token_id

    # -------------------------------
    # Tokenize test set for PPL
    # -------------------------------
    def tokenize_text(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            max_length=Config.MAX_LENGTH,
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

    # -------------------------------
    # Perplexity for base and fine-tuned
    # -------------------------------
    def compute_ppl(model, name: str) -> Dict[str, Any]:
        args = TrainingArguments(
            output_dir=os.path.join(Config.OUTPUT_DIR, f"eval_{name}"),
            per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
            seed=Config.SEED,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=args,
            eval_dataset=test_tokenized,
            data_collator=data_collator,
            processing_class=tokenizer,
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

    # -------------------------------
    # F1 on subset of test examples (10 examples)
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model.to(device)
    ft_model.to(device)
    base_model.eval()
    ft_model.eval()

    subset = test_data.select(range(10))
    examples_out: List[Dict[str, Any]] = []
    f1_base_list: List[float] = []
    f1_ft_list: List[float] = []

    for i, ex in enumerate(subset):
        pl = build_prompt_and_label(ex)
        prompt = pl["prompt"]
        gold = pl["label"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_ids_base = base_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
            gen_ids_ft = ft_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_text_base = tokenizer.decode(gen_ids_base[0], skip_special_tokens=True)
        full_text_ft = tokenizer.decode(gen_ids_ft[0], skip_special_tokens=True)

        
        if full_text_base.startswith(prompt):
            pred_base = full_text_base[len(prompt):].strip()
        else:
            pred_base = full_text_base

        if full_text_ft.startswith(prompt):
            pred_ft = full_text_ft[len(prompt):].strip()
        else:
            pred_ft = full_text_ft

        f1_base = word_f1(pred_base, gold)
        f1_ft = word_f1(pred_ft, gold)

        f1_base_list.append(f1_base)
        f1_ft_list.append(f1_ft)

        examples_out.append(
            {
                "index": i,
                "instruction": ex["instruction"],
                "input": ex["input"],
                "reference_output": gold,
                "outputs": {
                    "base": {
                        "model": Config.MODEL_NAME,
                        "strategy": "greedy",
                        "text": pred_base,
                        "f1": f1_base,
                    },
                    "fine_tuned": {
                        "model": f"{Config.MODEL_NAME}-lora",
                        "strategy": "greedy",
                        "text": pred_ft,
                        "f1": f1_ft,
                    },
                },
            }
        )
        print(
            f"[{i}] F1 base = {f1_base:.4f}, "
            f"F1 fine-tuned = {f1_ft:.4f}"
        )

    avg_f1_base = statistics.mean(f1_base_list)
    avg_f1_ft = statistics.mean(f1_ft_list)
    std_f1_base = statistics.pstdev(f1_base_list) if len(f1_base_list) > 1 else 0.0
    std_f1_ft = statistics.pstdev(f1_ft_list) if len(f1_ft_list) > 1 else 0.0

    print(
        f"Average F1 (base) over 10 examples: "
        f"{avg_f1_base:.4f} ± {std_f1_base:.4f}"
    )
    print(
        f"Average F1 (fine-tuned) over 10 examples: "
        f"{avg_f1_ft:.4f} ± {std_f1_ft:.4f}"
    )

    # -------------------------------
    # Save JSON: outputs/generations/test_set_evaluation.json
    # -------------------------------
    generations_dir = os.path.join(Config.OUTPUT_DIR, "generations")
    os.makedirs(generations_dir, exist_ok=True)

    out = {
        "config": {
            "seed": Config.SEED,
            "base_model": Config.MODEL_NAME,
            "adapter_path": Config.BEST_MODEL_DIR,
        },
        "metrics": {
            "base": {
                "perplexity": base_results["perplexity"],
                "eval_metrics": base_results["metrics"],
                "avg_f1": avg_f1_base,
                "std_f1": std_f1_base,
            },
            "fine_tuned": {
                "perplexity": ft_results["perplexity"],
                "eval_metrics": ft_results["metrics"],
                "avg_f1": avg_f1_ft,
                "std_f1": std_f1_ft,
            },
        },
        "examples": examples_out,
    }

    out_path = os.path.join(generations_dir, "test_set_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved test set evaluation to {out_path}")


if __name__ == "__main__":
    main()
