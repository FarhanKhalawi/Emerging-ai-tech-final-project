#sudo apt update
#sudo apt install -y python3 python3-venv python3-pip
#python3 -V                    # sanity check
#python3 -m venv .venv
#source .venv/bin/activate
#python -m pip install --upgrade pip
#pip install torch transformers evaluate rouge-score pandas
#python -m pip install --upgrade pip setuptools wheel
#pip install "bitsandbytes>=0.43.3" accelerate
#pip install scikit-learn

from datasets import load_dataset
import random
from transformers import AutoTokenizer

# -------------------------------
# 1. Load dataset
# -------------------------------
ds = load_dataset("yahma/alpaca-cleaned")

print(ds)
