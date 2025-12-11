import os
import json
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error
from peft import PeftModel

from config import Config


def load_model_and_tokenizer():

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

    model = PeftModel.from_pretrained(base_model, Config.BEST_MODEL_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    # -------------------------------
    # Checks, seeding, logging
    # -------------------------------
    if Config.HF_TOKEN is None:
        raise ValueError("HUGGINGFACE_HUB_TOKEN is not set in environment.")

    set_verbosity_error()

    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # -------------------------------
    # Define 10 instruction-style prompts
    # -------------------------------
    prompts: List[str] = [
        "Instruction: Explain why the sky is blue.\nResponse:",
        "Instruction: Give three tips for studying more effectively.\nResponse:",
        "Instruction: Describe a healthy breakfast for a busy student.\nResponse:",
        "Instruction: Summarize the main causes of climate change.\nResponse:",
        "Instruction: Explain the difference between RAM and ROM.\nResponse:",
        "Instruction: Suggest a weekly workout routine for beginners.\nResponse:",
        "Instruction: Describe how a blockchain works in simple terms.\nResponse:",
        "Instruction: Give advice to someone starting university.\nResponse:",
        "Instruction: Explain recursion to a first-year CS student.\nResponse:",
        "Instruction: Suggest a budget travel plan for a weekend in a European city.\nResponse:",
    ]

    results: List[Dict[str, Any]] = []

    # -------------------------------
    # Generate with 3 decoding strategies
    # -------------------------------
    for i, prompt in enumerate(prompts):
        print(f"Generating for prompt {i}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # --- Greedy decoding ---
        with torch.no_grad():
            greedy_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        greedy_full = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)
        greedy_text = (
            greedy_full[len(prompt):].strip()
            if greedy_full.startswith(prompt)
            else greedy_full
        )

        # --- Temperature sampling (0.7) ---
        with torch.no_grad():
            temp_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        temp_full = tokenizer.decode(temp_ids[0], skip_special_tokens=True)
        temp_text = (
            temp_full[len(prompt):].strip()
            if temp_full.startswith(prompt)
            else temp_full
        )

        # --- Top-p sampling (0.9) ---
        with torch.no_grad():
            topp_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        topp_full = tokenizer.decode(topp_ids[0], skip_special_tokens=True)
        topp_text = (
            topp_full[len(prompt):].strip()
            if topp_full.startswith(prompt)
            else topp_full
        )

        results.append(
            {
                "index": i,
                "instruction": prompt,
                "outputs": [
                    {
                        "model": f"{Config.MODEL_NAME}-lora",
                        "strategy": "greedy",
                        "text": greedy_text,
                    },
                    {
                        "model": f"{Config.MODEL_NAME}-lora",
                        "strategy": "temperature_0.7",
                        "text": temp_text,
                    },
                    {
                        "model": f"{Config.MODEL_NAME}-lora",
                        "strategy": "top_p_0.9",
                        "text": topp_text,
                    },
                ],
            }
        )

    # -------------------------------
    # Save JSON: outputs/generations/sampling_comparison.json
    # -------------------------------
    generations_dir = os.path.join(Config.OUTPUT_DIR, "generations")
    os.makedirs(generations_dir, exist_ok=True)

    out_path = os.path.join(generations_dir, "sampling_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "seed": Config.SEED,
                    "base_model": Config.MODEL_NAME,
                    "adapter_path": Config.BEST_MODEL_DIR,
                    "strategies": ["greedy", "temperature_0.7", "top_p_0.9"],
                },
                "examples": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved sampling comparison to {out_path}")


if __name__ == "__main__":
    main()
