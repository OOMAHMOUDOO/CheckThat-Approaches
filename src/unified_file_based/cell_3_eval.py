

import os
import json
import time
import re
import random
import threading
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
DATA_PATH = "/content/drive/MyDrive/CheckThat-Task2"
config = {
    "test_path": "/content/drive/MyDrive/CheckThat-Task2/clef_2026_final_english_test.json",
    "model_name": "Qwen/Qwen3-0.6B",
    "experiment_name": "clef-fact-check-grm-qwen3-06b-full-dataset-lora-16-en-unsloth",
    "output_dir": "/content/drive/MyDrive/CheckThat-checkpoints",
    "batch_size": 256,
    "max_length": 1024,
    "is_sanity": False, # Usually False for final test set
    "use_flash_attention": False,
    "use_dynamic_padding": True
}

SYSTEM_PROMPT = (
    "You are an expert fact-checking auditor. Your job is to evaluate whether a previous fact-checker's verdict and justification are correct given the claim and retrieved documents.\n\n"
    "You only respond with Yes or No, Yes if the Claim checker's Verdict and Justification are correct, No if they are incorrect\n /no_think"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utilities ---
def remove_label_pattern(text):
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "", text, flags=re.IGNORECASE
    ).strip()
    return text.replace("\n", " ")

class CustomClassifier(torch.nn.Module):
    def __init__(self, model_name, tokenizer, token=None, adapter_path=None, use_flash_attention=False):
        super().__init__()
        attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map="auto",
            token=token
        )
        self.yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = tokenizer.convert_tokens_to_ids("No")

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapters from {adapter_path}")
        else:
            print("Warning: No adapter path provided. Using base model.")

    def forward(self, input_ids, attention_mask):
        # Optimized Forward Pass: Extract only the hidden state of the last token
        # This prevents the "Logit Explosion" (saving ~10GB VRAM at large vocab/batch sizes)

        # Access the underlying transformer model (Qwen/Llama structure)
        # self.model is the PeftModel wrapping the ForCausalLM model
        base_model = self.model.get_base_model()

        # 1. Get hidden states from the backbone transformer
        transformer_outputs = base_model.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0] # [batch, seq_len, hidden_size]

        # 2. Extract ONLY the last token's hidden state
        last_token_hidden = hidden_states[:, -1, :] # [batch, hidden_size]

        # 3. Pass only that single hidden state through the LM head
        last_token_logits = base_model.lm_head(last_token_hidden) # [batch, vocab_size]

        # 4. Pull out our target Yes/No tokens
        target_logits = last_token_logits[:, [self.no_token_id, self.yes_token_id]]
        return target_logits

class VerifierEvaluator:
    def __init__(self, model, tokenizer, device, use_dynamic_padding=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_dynamic_padding = use_dynamic_padding
        self.model.eval()

    def score_batch(self, claims, verdicts, justifications, max_length=2048):
        texts = []
        for c, v, j in zip(claims, verdicts, justifications):
            input_text = f"Claim: {c}\nVerdict: {v}\nJustification: {j}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            texts.append(text)

        padding_strategy = True if self.use_dynamic_padding else "max_length"
        encoding = self.tokenizer(
            texts, truncation=True, padding=padding_strategy,
            max_length=max_length, return_tensors="pt"
        )
        ids = encoding["input_ids"].to(self.device)
        mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(ids, mask)
                return logits[:, 1].tolist()

def run_test_evaluation(evaluator, raw_data, batch_size=8):
    flat_inputs = []
    flat_indices = []
    for idx, sample in enumerate(raw_data):
        for t_idx, trace in enumerate(sample["Reasoning_traces"]):
            justification = remove_label_pattern(trace).split("Label:")[0].strip()
            flat_inputs.append({
                "claim": sample["claim"],
                "verdict": sample["Verdict_list"][t_idx],
                "justification": justification
            })
            flat_indices.append((idx, t_idx))

    all_scores = []
    batch_size_eval = config["batch_size"]
    for i in tqdm(range(0, len(flat_inputs), batch_size_eval), desc="Test Evaluation"):
        batch = flat_inputs[i:i+batch_size_eval]
        scores = evaluator.score_batch(
            [b["claim"] for b in batch],
            [b["verdict"] for b in batch],
            [b["justification"] for b in batch]
        )
        all_scores.extend(scores)

    results = [{"scores": []} for _ in range(len(raw_data))]
    for score, (sample_idx, _) in zip(all_scores, flat_indices):
        results[sample_idx]["scores"].append(score)

    predictions = []
    for idx, sample in enumerate(raw_data):
        s_list = results[idx]["scores"]
        #print(f"{len(sample['Reasoning_traces'])} and {len(s_list)}")
        best_trace_idx = np.argmax(s_list)
        best_verdict = sample["Verdict_list"][best_trace_idx]

        # Structure requested by user
        predictions.append({
            "query_id": idx,
            "Claim": sample["claim"],
            #"Label": best_verdict, # As requested: produce Label the same as Verdict_BoN
            "Verdict_BoN": best_verdict,
            "BoN_Verdict_list": sample["Verdict_list"],
            "Reasoning_traces": sample["Reasoning_traces"],
            "score_list": s_list
        })
    return predictions

# --- Timed Input Logic ---
def find_latest_checkpoint(output_dir, experiment_name):
    import glob
    # Look for folders matching experiment_name and containing "epoch_"
    pattern = os.path.join(output_dir, f"{experiment_name}*_epoch_*")
    folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]

    if not folders:
        # Try finding the base experiment directory as a fallback
        fallback = os.path.join(output_dir, experiment_name)
        return fallback if os.path.exists(fallback) else None

    # Extract epoch numbers and folder names for sorting
    def sort_key(folder_path):
        folder_name = os.path.basename(folder_path)
        match = re.search(r"epoch_(\d+)", folder_name)
        epoch_num = int(match.group(1)) if match else -1
        # Secondary sort by folder name (contains timestamp)
        return (epoch_num, folder_name)

    folders.sort(key=sort_key, reverse=True)
    return folders[0]

print("--- LoRA Checkpoint Selection (TEST SET) ---")
user_input_val = [None]
def get_input():
    user_input_val[0] = input("Enter path to LoRA checkpoint folder (or press Enter for latest): ").strip()

input_thread = threading.Thread(target=get_input)
input_thread.daemon = True
input_thread.start()
input_thread.join(timeout=20)

# Detect the latest checkpoint automatically
latest_checkpoint = find_latest_checkpoint(config["output_dir"], config["experiment_name"])
if latest_checkpoint:
    print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
else:
    print(f"Warning: No checkpoints found in {config['output_dir']}")
    latest_checkpoint = os.path.join(config["output_dir"], config["experiment_name"]) # Final fallback

if input_thread.is_alive():
    print(f"\nTimeout reached! Proceeding with: {latest_checkpoint}")
    chosen_adapter = latest_checkpoint
else:
    chosen_adapter = user_input_val[0] if user_input_val[0] else latest_checkpoint
    print(f"Using checkpoint: {chosen_adapter}")

# --- Execution ---
print(f"Loading base model and adapter: {chosen_adapter}")
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = CustomClassifier(
    config["model_name"], tokenizer,
    token=os.environ.get("HF_TOKEN"), adapter_path=chosen_adapter,
    use_flash_attention=config["use_flash_attention"]
)
model.to(device)
evaluator = VerifierEvaluator(model, tokenizer, device, use_dynamic_padding=config["use_dynamic_padding"])

print("Loading test data...")
with open(config["test_path"], "r") as f:
    test_raw = json.load(f)

if config["is_sanity"]:
    print("Sanity mode: using first 10 samples.")
    test_raw = test_raw[:10]

test_predictions = run_test_evaluation(evaluator, test_raw)

# Save results with random number
rand_num = random.randint(1000, 9999)
#output_filename = f"clef_predictions_test_{rand_num}_unsloth_qwen306_{chosen_adapter}.json"
chosen_adapter_name = chosen_adapter
if "/" in chosen_adapter_name:
  chosen_adapter_name = chosen_adapter_name.split("/")[-1]
output_path = f"/content/clef_predictions_test_{rand_num}_unsloth_qwen306_{chosen_adapter_name}_.json"
#os.makedirs(output_path, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(test_predictions, f, indent=4)

print(f"\nTest evaluation complete. Results saved to: {output_path}")
print(f"Produced {len(test_predictions)} predictions.")
