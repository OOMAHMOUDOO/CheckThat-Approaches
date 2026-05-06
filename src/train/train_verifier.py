# %% [markdown]
# # LoRA Fine-Tuning LLaMA-3B-Instruct
# This notebook fine-tunes a HuggingFace LLaMA-3B-Instruct model using LoRA (PEFT).

# %%
import os
import re
import json
import time
import random
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    get_cosine_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score
import evaluate

from peft import LoraConfig, get_peft_model


# %%
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
torch.set_default_dtype(torch.float32)


# %%
f1_metric = evaluate.load("f1")

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def remove_label_pattern(text):
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text.replace("\n", " ")


def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


# %%
from transformers import AutoModelForCausalLM

class CustomClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name,
        tokenizer,  # Pass tokenizer to get token IDs
        dropout_value=0.1,
        freeze_base_layer=True,
        use_lora=False,
        is_base_encoder=False, # Set to False for CausalLM
        lora_rank=8,
        lora_alpha=16,
    ):
        super().__init__()

        # 1. Use AutoModelForCausalLM instead of AutoModel
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.is_base_encoder = is_base_encoder
        
        # 2. Get token IDs for "Yes" and "No" (or target tokens)
        # Note: Llama-3 usually uses 9891 for "Yes" and 2201 for "No" 
        # but it is safer to derive them from the tokenizer.
        self.correct_token_id = tokenizer.convert_tokens_to_ids("CORRECT")
        self.incorrect_token_id = tokenizer.convert_tokens_to_ids("INCORRECT")

        if freeze_base_layer:
            for param in self.model.parameters():
                param.requires_grad = False

        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)

        # 3. Removed classification head as requested

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_token_logits = outputs.logits[:, -1, :] 
        
        # index 0: INCORRECT, index 1: CORRECT
        target_logits = last_token_logits[:, [self.incorrect_token_id, self.correct_token_id]]
        
        return target_logits



# %%
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe["input_text"].tolist()
        self.labels = dataframe["Class"].tolist()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        prompt = (
            "You are an expert fact-checking auditor. Your job is to evaluate whether a previous fact-checker's verdict and justification are correct given the claim and the raw evidence.\n\n"
            "You only respond with Yes or No, Yes if the Claim checker's Verdict and Justification are correct, No otherwise"
            f"{self.texts[idx]}\n\n"
            "[Evaluation]:"
        )
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        # CHANGE: Use torch.long for CrossEntropy targets
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# %%
class TrainerModule:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs,
        lr,
        patience,
        output_dir,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
        # CHANGE: Switch to CrossEntropyLoss
        self.loss_fn = torch.nn.CrossEntropyLoss()

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.05 * total_steps)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps,
            total_steps,
        )

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            self.model.train()

            total_loss, total_acc = 0, 0

            for batch in tqdm(self.train_loader):
                self.optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # model now returns (batch, 2)
                logits = self.model(input_ids, attention_mask)

                # CHANGE: logits is (batch, 2), labels is (batch) long
                loss = self.loss_fn(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

                # CHANGE: Use argmax for multi-class accuracy
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                total_acc += accuracy_score(labels.cpu().numpy(), preds)

            print(f"Train Loss: {total_loss / len(self.train_loader):.4f}")
            print(f"Train Acc: {total_acc / len(self.train_loader):.4f}")

            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss, total_acc = 0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                # CHANGE: Use argmax
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                total_acc += accuracy_score(labels.cpu().numpy(), preds)

        print(f"Val Loss: {total_loss / len(self.val_loader):.4f}")
        print(f"Val Acc: {total_acc / len(self.val_loader):.4f}")
        
        # Save logic
        tokenizer.save_pretrained(self.output_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.output_dir, f"model_epoch_{epoch}"),
        )


# %%
class VerifierEvaluator:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        base_model,
        use_decomp,
        device="cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_decomp = use_decomp

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = CustomClassifier(base_model, use_lora=True,is_base_encoder=False,)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def encode_input(self, claim, questions, verdict, justification, max_length=2048):
        input_text = f"Claim: {claim}\nVerdict: {verdict}\nJustification: {justification}"
        prompt = (
            "You are an expert fact-checking auditor. Your job is to evaluate whether a previous fact-checker's verdict and justification are correct given the claim and the raw evidence.\n\n"
            f"{input_text}\n\n"
            "[Evaluation]:"
        )

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return (
            encoding["input_ids"].to(self.device),
            encoding["attention_mask"].to(self.device),
        )

    def score(self, claim, questions, verdict, justification):
        ids, mask = self.encode_input(claim, questions, verdict, justification)
        with torch.no_grad():
            logits = self.model(ids, mask)
            # Use softmax to get probability of index 1 (CORRECT)
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, 1].item())


# %%
# ===== PATH PLACEHOLDERS =====
BASE_PATH = ""
TRAIN_JSON = BASE_PATH + "clef2026_gpt4_o_mini_val.json"
VAL_JSON = BASE_PATH + "clef2026_gpt4_o_mini_val.json"
TEST_JSON = BASE_PATH + "clef2026_gpt4_o_mini_test.json"

MAX_LENGTH = 2048
BATCH_SIZE = 12
EPOCHS = 2
LR = 1e-4
LORA_RANK = 128

BASE_MODEL = "llama-3.1-8b-instruct"
METHOD_NAME = "ce_loss_lora_ft"
MODEL_NAME = os.path.basename(BASE_MODEL.rstrip("/"))

OUTPUT_DIR = "./outputs/"+ MODEL_NAME + "_lora_" + str(LORA_RANK) + "_" + METHOD_NAME + "_lr_" + str(LR) + "_epochs_" + str(EPOCHS)

os.makedirs(OUTPUT_DIR, exist_ok=True)


with open(TRAIN_JSON, "r") as f:
    train_data = json.loads(f.read())


with open(VAL_JSON, "r") as f:
    val_data = json.loads(f.read())

with open(TEST_JSON, "r") as f:
    test_data = json.loads(f.read())

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

unrolled_train_data = unroll_data(train_data)
unrolled_val_data = unroll_data(val_data)
unrolled_test_data = unroll_data(test_data)

print(f"Original Train data size: {len(train_data)}")
print(f"Original Val data size: {len(val_data)}")
print(f"Original Test data size: {len(test_data)}")
print("-" * 30)
print(f"Unrolled Train data size: {len(unrolled_train_data)}")
print(f"Unrolled Val data size: {len(unrolled_val_data)}")
print(f"Unrolled Test data size: {len(unrolled_test_data)}")
train_df = pd.DataFrame(unrolled_train_data)
val_df = pd.DataFrame(unrolled_val_data)
test_df = pd.DataFrame(unrolled_test_data)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


train_dataset = TextDataset(train_df, tokenizer, MAX_LENGTH)
val_dataset = TextDataset(val_df, tokenizer, MAX_LENGTH)
test_dataset = TextDataset(test_df, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = CustomClassifier(
    BASE_MODEL,
    tokenizer=tokenizer,
    use_lora=True,
    is_base_encoder=False,
    lora_rank=LORA_RANK,
    lora_alpha=LORA_RANK*2
)

print_trainable_parameters(model)

trainer = TrainerModule(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    lr=LR,
    patience=2,
    output_dir=OUTPUT_DIR,
)

trainer.train()


# %% [markdown]
# 

# %%
with open(VAL_CSV, "r") as f:
  test_data = json.load(f)

predictions = []

evaluator = VerifierEvaluator(
    model_path=f"{OUTPUT_DIR}/model_epoch_3",
    tokenizer_path=BASE_MODEL,
    base_model=BASE_MODEL,
    use_decomp=True,
)

# Example scoring
for idx, sample in enumerate(test_data):
  verdict_list = []
  verifier_score_list = []
  justification_list = []
  approved_majority_list = []
  for decoding_sample in range(1, len(sample["Reasoning_traces"]) + 1):
    justification = (
        remove_label_pattern(
            sample["Reasoning_traces"][decoding_sample - 1]
        ).split("Label:")[0]
    )
    score = evaluator.score(
    sample["claim"],
    sample.get("Questions", ""),
    sample["Verdict_list"][decoding_sample - 1].lower(),
    justification,
)
    verdict_list.append(sample["Verdict_list"][decoding_sample - 1])
    justification_list.append(justification)
    verifier_score_list.append(score)
    best_verdict = verdict_list[np.argmax(np.array(verifier_score_list))]
    predictions.append({
        "query_id": idx,
        "Claim": sample["claim"],
        "Label": sample["label"],
        "Verdict_BoN": best_verdict,
        "BoN_Verdict_list": verdict_list,
        "Reasoning_traces": justification_list,
        "score_list": verifier_score_list,
    })




# %%
with open(f"{OUTPUT_DIR}/clef_predictions.json", "w") as fp:
  json.dump(predictions, fp, indent=4)

# %%



