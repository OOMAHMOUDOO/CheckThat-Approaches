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
import pandas as pd
import random
import torch
from tqdm.auto import tqdm
from transformers import TrainerCallback
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
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
    "train_path": "/workspace/CheckThat-Task2/deepseek_audit_results_20k.jsonl.",
    "test_path": "/workspace/CheckThat-Task2/clef_2026_final_english_test.json",
    "model_name": "Qwen/Qwen3-4B",
    "lora_rank": 64,
    "experiment_name": "clef-distill-ds-lr-64-qwen3-4b",
    "is_sanity": True,  # Set to False for full run
    "batch_size": 64, # Better balance for CPU/GPU throughput
    "gradient_accumulation_steps": 1, 
    "epochs": 2,
    "lr": 1e-5,
    "max_length": 1024, # Reduced from 2048 for 2x-4x speedup (if data fits)
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
SYSTEM_PROMPT = (
    "You are an expert fact-checking auditor specializing in numerical, statistical, and temporal claims. "
    "Your task is to rigorously evaluate whether a previous fact-checker's verdict and justification are correct, "
    "given the original claim and retrieved evidence.\n\n"
    "Verdict Definitions:\n"
    "- True: The evidence fully supports the claim.\n"
    "- False: The evidence directly contradicts the claim.\n"
    "- Conflicting: The evidence contains contradictory elements or is inconclusive.\n\n"
    "Evaluation Criteria:\n"
    "1. Evidence Grounding: Does the justification accurately reflect the provided evidence without hallucination?\n"
    "2. Numerical/Temporal Precision: Are all quantities, dates, and calculations correctly verified? Beware of the 'Numeric-Truth Effect'.\n"
    "3. Logical Consistency: Does the reasoning chain logically support the verdict?\n"
    "4. Verdict Alignment: Is the stated verdict consistent with the definitions above given the evidence?\n\n"
    "Output Rules:\n"
    "- Think step-by-step inside <think>... </think> tags.\n"
    "- After </think>, output EXACTLY one of the following formats:\n"
    "  • Yes, verdict is correct.\n"
    "  • No, verdict should be [Correct Verdict].\n"
    "- Replace [Correct Verdict] with True, False, or Conflicting.\n"
    "- Do not output any additional text after this line."
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# --- Data Loading ---
print("Loading processed data...")
with open(config["train_path"], "r") as f: 
    train_data = f.readlines()
    train_data = [json.loads(row) for row in train_data]

if config["is_sanity"]:
    print("Sanity mode: using first few samples.")
    train_data = train_data[:10]

train_df = pd.DataFrame(train_data)
print(f"Loaded samples - Train: {len(train_df)}")

# Create HF Datasets
train_dataset = Dataset.from_pandas(train_df)

# --- Model & Tokenizer Setup ---
print("Initializing model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_name"],
    max_seq_length=config["max_length"],
    load_in_4bit=False,
    dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN")
)

model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_rank"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=config["lora_rank"] * 2,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Apply Chat Template
# Chat template is handled by the model's default tokenizer configuration

def formatting_prompts_func(examples):
    prompts = examples["input_text"]
    completions = examples["response"]
    texts = []
    for p, c in zip(prompts, completions):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
            {"role": "assistant", "content": c}
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

print("Formatting datasets...")
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)

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

print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_ratio=0.05,
        num_train_epochs=config["epochs"],
        learning_rate=float(config["lr"]),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=1000,
        output_dir=output_dir_full,
        dataloader_num_workers=4,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        max_seq_length=config["max_length"],
        dataset_num_proc=2,
        packing=False,
        assistant_only_loss=True,
    ),
    train_dataset=train_dataset,
    processing_class=tokenizer,
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
        model.push_to_hub(repo_id, token=token)
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"Successfully uploaded to Hub: {repo_id}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")

print("\nTraining Phase Finished.")

