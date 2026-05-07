# --- UNIFIED INSTALLATION (Run this in a separate cell FIRST) ---
# !pip uninstall unsloth -y
# !pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-cache-dir trl peft transformers accelerate spacy nltk
# !python -m spacy download en_core_web_sm
import os
import json
import time
import datetime
import re
import numpy as np

import random
import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer
from huggingface_hub import HfApi, login


# --- Environment Setup ---
os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
# If you have HF_TOKEN in Colab Secrets, it will be used.
# Otherwise, you might need to login manually:
# from huggingface_hub import login
# login()

DATA_PATH = "/content/drive/MyDrive/CheckThat-Task2"

# Configuration
config = {
    "train_path": "/workspace/CheckThat-Task2/deepseek_audit_results_20k_conversational.jsonl",
    "test_path": "/workspace/CheckThat-Task2/clef_2026_final_english_test.json",
    "model_name": "Qwen/Qwen3-4B",
    "lora_rank": 64,
    "experiment_name": "clef-distill-ds-lr-64-qwen3-4b",
    "is_sanity": True,  # Set to False for full run
    "batch_size": 64, # Better balance for CPU/GPU throughput
    "gradient_accumulation_steps": 1, 
    "epochs": 2,
    "lr": 1e-5,
    "max_length": 2048, # Reduced from 2048 for 2x-4x speedup (if data fits)
    "output_dir": "/workspace/CheckThat-checkpoints",
    "use_flash_attention": False, # Disabled as per user request
    "use_dynamic_padding": True
}

if config.get("is_sanity"):
    config["experiment_name"] += "-sanity"
    config["epochs"] += 1

# Login to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged into HuggingFace")

# Unified System Prompt




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# --- Data Loading ---
print("Loading processed data using HF datasets...")
train_dataset = load_dataset("json", data_files=config["train_path"], split="train")

if config["is_sanity"]:
    print("Sanity mode: using first few samples.")
    train_dataset = train_dataset.select(range(min(10, len(train_dataset))))

print(f"Loaded samples - Train: {len(train_dataset)}")

# Filter dataset to ensure every example has an assistant response
# This prevents RuntimeError when using assistant_only_loss=True
def has_assistant(example):
    return any(m.get("role") == "assistant" for m in example.get("messages", []))

train_dataset = train_dataset.filter(has_assistant)
print(f"Filtered samples - Train: {len(train_dataset)}")

# --- Model & Tokenizer Setup ---
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for training

# PEFT Configuration
peft_config = LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_rank"] * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply Chat Template
# Chat template is handled by the model's default tokenizer configuration






# --- Trainer Setup ---
output_dir_full = os.path.join(config["output_dir"], config["experiment_name"])
os.makedirs(output_dir_full, exist_ok=True)

class PushLoRACallback(TrainerCallback):
    def __init__(self, hf_token, base_repo_name, save_steps):
        self.hf_token = hf_token
        self.base_repo_name = base_repo_name
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.save_steps == 0:
            model = kwargs.get("model")
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            repo_id = f"{self.base_repo_name}-step-{step}"
            print(f"\n[PushLoRACallback] Pushing LoRA adapters to {repo_id}...")
            try:
                model.push_to_hub(repo_id, token=self.hf_token)
                if tokenizer:
                    tokenizer.push_to_hub(repo_id, token=self.hf_token)
                print(f"[PushLoRACallback] Successfully pushed to {repo_id}!\n")
            except Exception as e:
                print(f"\n[PushLoRACallback] Error pushing to Hub: {e}\n")



callbacks_list = [
    PushLoRACallback(hf_token=os.environ.get('HF_TOKEN'), base_repo_name=config["experiment_name"], save_steps=1000)
]


# --- Modern SFT Setup ---
sft_args = SFTConfig(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["lr"],
    num_train_epochs=config["epochs"],
    max_length=config["max_length"],
    
    # THE REPLACEMENT FOR DataCollatorForCompletionOnlyLM
    # This automatically masks user/system tokens in the loss
    assistant_only_loss=True, 
    
    # Precision settings
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    optim="adamw_torch_fused", # Faster for 4B+ models
    report_to="none",
)

print("Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=config["model_name"],
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=sft_args,
    callbacks=callbacks_list
)

print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(os.path.join(output_dir_full, "final_model"))

token = os.environ.get('HF_TOKEN')
if token:
    print("Uploading to Hugging Face Hub...")
    try:
        api = HfApi()
        repo_id = f"{api.whoami(token=token)['name']}/{config['experiment_name']}"
        trainer.push_to_hub(repo_id, token=token)
        print(f"Successfully uploaded to Hub: {repo_id}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")

print("\nTraining Phase Finished.")

