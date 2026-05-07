# --- UNIFIED INSTALLATION (Run this in a separate cell FIRST) ---
# !pip uninstall unsloth -y
# !pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-cache-dir trl peft transformers accelerate

import os
import json
import time
import datetime
import re
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler # More robust scheduler import
)
from peft import LoraConfig, get_peft_model
from unsloth import FastLanguageModel # Added Unsloth
from sklearn.metrics import accuracy_score
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
    "train_path": "/content/clef_2026_checkthat_english_train.json",
    "val_path": "/content/drive/MyDrive/CheckThat-Task2/clef2026_gpt4_o_mini_val_english.json",
    "test_path": "/content/drive/MyDrive/CheckThat-Task2/clef_2026_final_english_test.json",
    "model_name": "Qwen/Qwen3-4B",
    "lora_rank": 16,
    "experiment_name": "clef-fact-check-grm-qwen3-4b-full-dataset-lora-8-en-unsloth",
    "is_sanity": False,  # Set to False for full run
    "batch_size": 64, # Better balance for CPU/GPU throughput
    "gradient_accumulation_steps": 1, 
    "epochs": 2,
    "lr": 1e-5,
    "max_length": 1024, # Reduced from 2048 for 2x-4x speedup (if data fits)
    "output_dir": "/content/drive/MyDrive/CheckThat-checkpoints",
    "use_flash_attention": False, # Disabled as per user request
    "use_dynamic_padding": True
}

# Login to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged into HuggingFace")

# Unified System Prompt
SYSTEM_PROMPT = (
    "You are an expert fact-checking auditor. Your job is to evaluate whether a previous fact-checker's verdict and justification are correct given the claim and retrieved documents.\n\n"
    "You only respond with Yes or No, Yes if the Claim checker's Verdict and Justification are correct, No if they are incorrect\n /no_think"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Utilities ---
def remove_label_pattern(text):
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()
    return text.replace("\n", " ")

def unroll_data(data):
    unrolled = []
    for idx, item in enumerate(data):
        label = item.get("label", "")
        claim = item.get("claim", "")
        verdict_list = item.get("Verdict_list", [])
        reasoning_traces = item.get("Reasoning_traces", [])
        sample_id = item.get("id", idx)

        for v, trace in zip(verdict_list, reasoning_traces):
            justification = remove_label_pattern(trace).split("Label:")[0].strip()
            class_label = 1 if str(v).lower() == str(label).lower() else 0

            unrolled.append({
                "sample_id": sample_id,
                "input_text": f"Claim: {claim}\nVerdict: {v}\nJustification: {justification}",
                "Label": label,
                "Verdict": v,
                "Class": class_label
            })
    return unrolled

# --- Data Loading ---
print("Loading data...")
with open(config["train_path"], "r") as f: train_raw = json.load(f)
with open(config["val_path"], "r") as f: val_raw = json.load(f)

if config["is_sanity"]:
    print("Sanity mode: using first few samples.")
    train_raw = train_raw[:10]
    val_raw = val_raw[:10]

train_unrolled = unroll_data(train_raw)
val_unrolled = unroll_data(val_raw)
train_df = pd.DataFrame(train_unrolled)
val_df = pd.DataFrame(val_unrolled)
print(f"Unrolled samples - Train: {len(train_df)}, Val: {len(val_df)}")

# --- Model & Dataset ---
class CustomClassifier(torch.nn.Module):
    def __init__(self, model_name, lora_rank=16, lora_alpha=32, max_length=2048, token=None):
        super().__init__()
        
        # 1. Load Unsloth FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_length,
            load_in_4bit = False, # Disabled to skip dequantization overhead
            dtype = torch.bfloat16, # Use pure bfloat16 for max H100 Tensor Core speed
            token = token,
        )
        
        # 2. Add LoRA adapters using Unsloth's optimized method
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = lora_rank,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = lora_alpha,
            lora_dropout = 0, # Optimized for 0
            bias = "none",    # Optimized for "none"
            use_gradient_checkpointing = "unsloth", # Even faster checkpointing
            random_state = 3407,
        )

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        # Correct navigation for Unsloth/PEFT structure:
        # self.model is the PeftModel
        # self.model.get_base_model() is the Qwen2ForCausalLM
        # self.model.get_base_model().model is the Qwen2Model (Backbone)
        base_model = self.model.get_base_model()
        
        # 1. Get hidden states from the backbone transformer
        transformer_outputs = base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0] # [batch, seq_len, hidden_size]
        
        # 2. Extract ONLY the last token's hidden state (shape: [batch, 2560])
        last_token_hidden = hidden_states[:, -1, :] 
        
        # 3. Pass through the LM head to get logits (shape: [batch, 151936])
        last_token_logits = base_model.lm_head(last_token_hidden) 
        
        # 4. Pull out our target Yes/No tokens
        target_logits = last_token_logits[:, [self.no_token_id, self.yes_token_id]]
        return target_logits

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, use_dynamic_padding=True):
        self.labels = dataframe["Class"].tolist()
        texts = dataframe["input_text"].tolist()
        self.encodings = []
        
        padding_strategy = True if use_dynamic_padding else "max_length"
        
        print(f"Pre-tokenizing {len(texts)} samples...")
        for text in texts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            encoding = tokenizer(
                prompt, truncation=True, padding=padding_strategy,
                max_length=max_length, return_tensors="pt"
            )
            self.encodings.append({k: v.squeeze(0) for k, v in encoding.items()})

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# --- Trainer ---
class TrainerModule:
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Use Fused AdamW for 10-15% speed boost on H100
        self.optimizer = AdamW(model.parameters(), lr=config["lr"], eps=1e-8, fused=True)
        self.grad_acc_steps = config.get("gradient_accumulation_steps", 1)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        total_steps = len(train_loader) * config["epochs"]
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps
        )
        self.api = HfApi()
        self.exp_name = config["experiment_name"]
        self.output_dir = os.path.join(config["output_dir"], self.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.exp_name}_{timestamp}_epoch_{epoch}"
        save_path = os.path.join(self.config["output_dir"], checkpoint_name)
        os.makedirs(save_path, exist_ok=True)

        print(f"Saving checkpoint to local Drive: {save_path}")
        # Unsloth models save standard Peft-compatible adapters
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        return save_path

    def upload_to_hf(self, epoch, local_path):
        try:
            repo_id = f"{self.api.whoami(token=os.environ.get('HF_TOKEN'))['name']}/{self.exp_name}-epoch-{epoch}"
            print(f"Uploading checkpoint for epoch {epoch} to Hub: {repo_id}")
            # Push the Unsloth/Peft adapters
            self.model.push_to_hub(repo_id, token=os.environ.get("HF_TOKEN"))
            print(f"Successfully uploaded to Hub: {repo_id}")
        except Exception as e:
            print(f"Skipping HF upload: {e}")

    def train(self):
        for epoch in range(self.config["epochs"]):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            self.model.train()
            total_loss, total_acc = 0, 0
            self.optimizer.zero_grad()

            for i, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                ids, mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["labels"].to(self.device)

                # Use autocast for mixed precision safety
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(ids, mask)
                    loss = self.loss_fn(logits, labels)
                    loss = loss / self.grad_acc_steps

                loss.backward()

                if (i + 1) % self.grad_acc_steps == 0:
                    # Added gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.grad_acc_steps
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                total_acc += accuracy_score(labels.cpu().numpy(), preds)

            print(f"Train Loss: {total_loss/len(self.train_loader):.4f} | Train Acc: {total_acc/len(self.train_loader):.4f}")
            self.evaluate(epoch)

            # Save checkpoint locally first
            local_path = self.save_checkpoint(epoch)
            # Then optionally upload
            self.upload_to_hf(epoch, local_path)

            # Clear cache
            torch.cuda.empty_cache()

    def evaluate(self, epoch):
        self.model.eval()
        total_loss, total_acc = 0, 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                ids, mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["labels"].to(self.device)
                logits = self.model(ids, mask)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                total_acc += accuracy_score(labels.cpu().numpy(), preds)
        print(f"Val Loss: {total_loss/len(self.val_loader):.4f} | Val Acc: {total_acc/len(self.val_loader):.4f}")

# Clear existing memory
if 'model' in locals(): del model
if 'trainer' in locals(): del trainer
torch.cuda.empty_cache()

# --- Execution ---
# Initializing Unsloth model
model_wrapper = CustomClassifier(
    config["model_name"],
    lora_rank=config["lora_rank"],
    lora_alpha=config["lora_rank"]*2,
    max_length=config["max_length"],
    token=os.environ.get("HF_TOKEN")
)
model = model_wrapper
tokenizer = model_wrapper.tokenizer

# Use a data collator for dynamic padding if enabled
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) if config["use_dynamic_padding"] else None

train_dataset = TextDataset(train_df, tokenizer, config["max_length"], use_dynamic_padding=config["use_dynamic_padding"])
val_dataset = TextDataset(val_df, tokenizer, config["max_length"], use_dynamic_padding=config["use_dynamic_padding"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=data_collator,
    num_workers=16, # Increased to 16 since you have 48 cores!
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    collate_fn=data_collator,
    num_workers=16,
    pin_memory=True
)

trainer = TrainerModule(model, train_loader, val_loader, tokenizer, config)
trainer.train()
print("\nTraining Phase Finished.")

