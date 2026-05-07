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
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    "dataset_name": "Mahmoud669401/my-ds-clef-20-chat-dataset", # Updated as per recent upload
    "model_name": "Qwen/Qwen3-4B-Instruct",
    "lora_rank": 16,
    "experiment_name": "qwen3-4b-sft-clef-cont",
    "is_sanity": False, 
    "batch_size": 1,
    "gradient_accumulation_steps": 8, 
    "epochs": 1,
    "lr": 2e-4,
    "max_length": 2048,
    "output_dir": "./qwen3-4b-sft",
    "resume_checkpoint": "Mahmoud669401/meta-3b-sft-clef-20k", # e.g., "username/checkpoint-name"
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
print(f"Loading dataset from Hub: {config['dataset_name']}")
dataset = load_dataset(config["dataset_name"], split="train")

if config["is_sanity"]:
    print("Sanity mode: using first few samples.")
    dataset = dataset.select(range(min(10, len(dataset))))

print(f"Loaded samples: {len(dataset)}")

# Filter dataset to ensure every example has an assistant response
# This prevents RuntimeError when using assistant_only_loss=True
def has_assistant(example):
    return any(m.get("role") == "assistant" for m in example.get("messages", []))

train_dataset = dataset.filter(has_assistant)
print(f"Filtered samples - Train: {len(train_dataset)}")

# --- Model & Tokenizer Setup ---
print(f"Loading tokenizer and model: {config['model_name']}")
tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"], 
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for training

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# PEFT Configuration
peft_config = LoraConfig(
    r=config["lora_rank"],
    lora_alpha=config["lora_rank"] * 2,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    task_type="CAUSAL_LM",
)

# --- Handle Resuming from Checkpoint ---
if config.get("resume_checkpoint"):
    print(f"Loading existing LoRA adapter from: {config['resume_checkpoint']}")
    model = PeftModel.from_pretrained(
        model,
        config["resume_checkpoint"],
        is_trainable=True,
        token=hf_token
    )
    sft_peft_config = None  # Trainer doesn't need peft_config if model is already PeftModel
else:
    sft_peft_config = peft_config

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

class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, log_path="inference_training.log", steps=100):
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.steps = steps
        # Sample 1 item to track
        self.samples = dataset.select(range(min(1, len(dataset))))

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            tokenizer = self.tokenizer
            model.eval()
            
            print(f"\n--- Step {state.global_step} Periodic Inference ---")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Step {state.global_step} ---\n")
                
                for i, example in enumerate(self.samples):
                    # Filter messages to exclude the final assistant response we're training on
                    messages = [m for m in example["messages"] if m["role"] in ["system", "user"]]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            stop_strings=["</think>"],
                            tokenizer=tokenizer,
                        )
                    
                    prompt_len = inputs["input_ids"].shape[1]
                    gen_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
                    
                    out_str = f"\nSample {i}:\nPrompt Snippet: ...{prompt[-150:]}\nResponse: {gen_text}\n{'-'*40}"
                    print("\033[31m" + out_str+ "\033[0m")
                    f.write(out_str)
            
            model.train()

callbacks_list = [
    PushLoRACallback(hf_token=os.environ.get('HF_TOKEN'), base_repo_name=config["experiment_name"], save_steps=1000),
    InferenceCallback(tokenizer=tokenizer, dataset=train_dataset, steps=100)
]


# --- Modern SFT Setup ---
sft_args = SFTConfig(
    output_dir=config["output_dir"],
    per_device_train_batch_size=config["batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["lr"],
    num_train_epochs=config["epochs"],
    logging_steps=10,

    # important
    assistant_only_loss=True,

    # recommended
    bf16=True,
    packing=True,
    max_seq_length=config["max_length"],
    
    save_strategy="steps",
    save_steps=500,
    optim="adamw_torch_fused",
    report_to="none",
)

print("Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=sft_peft_config,
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

