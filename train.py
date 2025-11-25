#sudo apt update
#sudo apt install -y python3 python3-venv python3-pip
#python3 -V                    # sanity check
#python3 -m venv .venv
#source .venv/bin/activate
#python -m pip install --upgrade pip
#pip install torch transformers evaluate rouge-score pandas
#python -m pip install --upgrade datasets huggingface_hub --break-system-packages
#python -m pip install transformers peft accelerate --break-system-packages


from datasets import load_dataset
import random
from transformers import AutoTokenizer

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = load_dataset("yahma/alpaca-cleaned")

print(dataset)

SEED = 42
random.seed(SEED)

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

# -------------------------------
# 3. Format train/val/test
# -------------------------------
train_data = train_data.map(format_example)
val_data   = val_data.map(format_example)
test_data  = test_data.map(format_example)

# Print one example to check
print("Example formatted prompt:")
print(train_data[0]["prompt"])
print("-----")
print("Label:")
print(train_data[0]["label"])
