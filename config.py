# config.py
import os

class Config:

    SEED = 42

    
    HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

    
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    DATASET_NAME = "yahma/alpaca-cleaned"

    
    TRAIN_SAMPLES = 10_000
    VAL_SAMPLES   = 2_000
    TEST_SAMPLES  = 2_000

    
    MAX_LENGTH = 512

    
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

   
    USE_FP16 = True  

    
    OUTPUT_DIR      = "./outputs"
    CHECKPOINT_DIR  = "./outputs/checkpoints"
    BEST_MODEL_DIR  = "./outputs/best_model.pt"   
    PLOTS_DIR       = "./outputs/plots"

    
    LORA_R         = 8
    LORA_ALPHA     = 16
    LORA_DROPOUT   = 0.1
    LORA_TASK_TYPE = "CAUSAL_LM"
