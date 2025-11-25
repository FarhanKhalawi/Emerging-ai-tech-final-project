from datasets import load_dataset
import random
from transformers import AutoTokenizer

SEED = 42

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
# 4. Load tokenizer
# -------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

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
