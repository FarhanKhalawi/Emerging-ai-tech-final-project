# config.py
import os

class Config:
    # Random seed
    SEED = 42

    # Hugging Face token (from environment)
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Model and dataset
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "yahma/alpaca-cleaned"

    # Data split sizes
    TRAIN_SAMPLES = 10_000
    VAL_SAMPLES   = 2_000
    TEST_SAMPLES  = 2_000

    # Tokenization
    MAX_LENGTH = 512

    # Training hyperparameters
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE  = 4
    LEARNING_RATE = 2e-4
    GRADIENT_ACCUMULATION_STEPS = 4
    WARMUP_STEPS = 100
    NUM_EPOCHS   = 3

    LOGGING_STEPS = 50
    EVAL_STEPS    = 200
    SAVE_STEPS    = 200
    SAVE_TOTAL_LIMIT = 3

    # Mixed precision
    USE_FP16 = True  # will only activate if CUDA is available

    # Output directories
    OUTPUT_DIR      = "./outputs"
    CHECKPOINT_DIR  = "./outputs/checkpoints"
    BEST_MODEL_DIR  = "./outputs/best_model.pt"   # directory name with .pt
    PLOTS_DIR       = "./outputs/plots"

    # LoRA config
    LORA_R         = 8
    LORA_ALPHA     = 16
    LORA_DROPOUT   = 0.1
    LORA_TASK_TYPE = "CAUSAL_LM"
