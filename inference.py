import os
import json
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SEED = 42
MODEL_DIR = "./outputs/best_model.pt"

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HUGGINGFACE_HUB_TOKEN is not set.")


def load_model_and_tokenizer():
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
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 10 example instructions (you can edit these)
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

    for i, prompt in enumerate(prompts):
        print(f"Generating for prompt {i}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Greedy
        with torch.no_grad():
            greedy_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        greedy_text = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

        # Temperature 0.7
        with torch.no_grad():
            temp_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        temp_text = tokenizer.decode(temp_ids[0], skip_special_tokens=True)

        # Top-p 0.9
        with torch.no_grad():
            topp_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        topp_text = tokenizer.decode(topp_ids[0], skip_special_tokens=True)

        results.append(
            {
                "index": i,
                "instruction": prompt,
                "outputs": [
                    {
                        "model": "llama-3.2-1B-lora",
                        "strategy": "greedy",
                        "text": greedy_text,
                    },
                    {
                        "model": "llama-3.2-1B-lora",
                        "strategy": "temperature_0.7",
                        "text": temp_text,
                    },
                    {
                        "model": "llama-3.2-1B-lora",
                        "strategy": "top_p_0.9",
                        "text": topp_text,
                    },
                ],
            }
        )

    os.makedirs("./outputs/generations", exist_ok=True)
    out_path = "./outputs/generations/sampling_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sampling comparison to {out_path}")


if __name__ == "__main__":
    main()
