# -*- coding: utf-8 -*-
"""
Cell 2: Evaluation Phase
Asks for LoRA checkpoint path, waits 50s, then runs eval on test set.
"""

import os
import json
import time
import re
import random
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import classification_report
import spacy
import nltk
from nltk import pos_tag, word_tokenize
from collections import defaultdict

# Setup NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)

# --- Configuration (Should match Cell 1) ---
DATA_PATH = "/content/drive/MyDrive/CheckThat-Task2"
config = {
    "test_path": "/content/drive/MyDrive/CheckThat-Task2/clef2026_gpt4_o_mini_val_english.json",
    "model_name": "Qwen/Qwen3-0.6B",
    "experiment_name": "clef-fact-check-grm-qwen3-06b-full-dataset-lora-16-en-unsloth",
    "output_dir": "/content/drive/MyDrive/CheckThat-checkpoints",
    "batch_size": 256,
    "max_length": 1024,
    "is_sanity": False, # Set to False for full evaluation
    "use_flash_attention": False,
    "use_dynamic_padding": True
}

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

USER_TEMPLATE = (
    "### Claim:\n{claim}\n\n"
    "### Extracted Context (NER & Verbs):\nPersons: {persons}\nPlaces: {places}\nVerbs: {verbs}\n\n"
    "### Retrieved Evidence:\n{evidence}\n\n"
    "### Previous Verdict:\n{verdict}\n\n"
    "### Previous Justification:\n{justification}\n\n"
    "Audit this verdict and justification according to the criteria and definitions. "
    "Provide your reasoning in <think> tags, then output your final assessment in the exact required format."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spacy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model en_core_web_sm...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities_and_verbs(text):
    if not text:
        return "None", "None", "None"
    doc = nlp(text)
    persons = set([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    places = set([ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")])
    
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    verbs = set([word for word, tag in tags if tag.startswith('VB')])
    
    return ", ".join(persons) or "None", ", ".join(places) or "None", ", ".join(verbs) or "None"

def remove_label_pattern(text):
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "", text, flags=re.IGNORECASE
    ).strip()
    return text.replace("\n", " ")

class ProbabilisticEvaluator:
    def __init__(self, model, tokenizer, device, use_dynamic_padding=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_dynamic_padding = use_dynamic_padding
        self.model.eval()

    def score_candidates_batch(self, batch_prompts, batch_candidates):
        texts = [p + c for p, c in zip(batch_prompts, batch_candidates)]
        
        padding_strategy = True if self.use_dynamic_padding else "max_length"
        encoding = self.tokenizer(
            texts, truncation=True, padding=padding_strategy,
            max_length=2048, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        prompt_encoding = self.tokenizer(batch_prompts, padding=False, truncation=True)
        prompt_lengths = [len(ids) for ids in prompt_encoding["input_ids"]]
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(shift_labels.size()) 
        
        scores = []
        for i in range(len(batch_prompts)):
            p_len = prompt_lengths[i]
            valid_len = shift_mask[i].sum().item()
            
            if valid_len <= p_len - 1:
                scores.append(-9999.0)
                continue
                
            candidate_loss = losses[i, p_len-1:valid_len].sum().item()
            scores.append(-candidate_loss)
            
        return scores

def run_full_evaluation(evaluator, tokenizer, raw_data, batch_size=8):
    flat_prompts = []
    flat_candidates = []
    flat_meta = []
    
    for idx, sample in enumerate(tqdm(raw_data, desc="Preparing Prompts")):
        claim = sample.get("claim", "")
        evidence = sample.get("evidence", sample.get("documents", "No evidence provided."))
        combined_text = f"{claim} {evidence}"
        persons, places, verbs = extract_entities_and_verbs(combined_text)
        
        for t_idx, trace in enumerate(sample["Reasoning_traces"]):
            justification = remove_label_pattern(trace).split("Label:")[0].strip()
            v = sample["Verdict_list"][t_idx].lower()
            
            user_input = USER_TEMPLATE.format(
                claim=claim,
                persons=persons,
                places=places,
                verbs=verbs,
                evidence=evidence,
                verdict=v.capitalize(),
                justification=justification
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            c_yes = f"<think>\nThe previous justification logic correctly reflects the evidence to conclude {v.capitalize()}.\n</think>\nYes, verdict is correct."
            candidates = [(c_yes, v)]
            
            for possible_label in ["true", "false", "conflicting"]:
                if possible_label != v:
                    c_no = f"<think>\nThe previous justification is flawed. The evidence actually supports the conclusion of {possible_label.capitalize()}.\n</think>\nNo, verdict should be {possible_label.capitalize()}."
                    candidates.append((c_no, possible_label))
            
            for c_text, c_label in candidates:
                flat_prompts.append(prompt_text)
                flat_candidates.append(c_text)
                flat_meta.append({
                    "sample_idx": idx,
                    "trace_idx": t_idx,
                    "cand_label": c_label,
                })

    print("Running Probabilistic Evaluation...")
    all_scores = []
    for i in tqdm(range(0, len(flat_prompts), batch_size), desc="Scoring Batches"):
        b_prompts = flat_prompts[i:i+batch_size]
        b_cands = flat_candidates[i:i+batch_size]
        scores = evaluator.score_candidates_batch(b_prompts, b_cands)
        all_scores.extend(scores)

    trace_scores = defaultdict(list)
    for score, meta in zip(all_scores, flat_meta):
        s_idx = meta["sample_idx"]
        t_idx = meta["trace_idx"]
        trace_scores[(s_idx, t_idx)].append({
            "label": meta["cand_label"],
            "score": score
        })
        
    sample_votes = [{"true": 0.0, "false": 0.0, "conflicting": 0.0} for _ in range(len(raw_data))]
    sample_scores_list = [[] for _ in range(len(raw_data))]
    
    for (s_idx, t_idx), scores_list in trace_scores.items():
        scores_tensor = torch.tensor([item["score"] for item in scores_list])
        probs = torch.softmax(scores_tensor, dim=0).tolist()
        
        trace_vote_sum = 0
        for i, item in enumerate(scores_list):
            label = item["label"]
            prob = probs[i]
            sample_votes[s_idx][label] += prob
            if i == 0: 
                trace_vote_sum = prob
                
        sample_scores_list[s_idx].append(trace_vote_sum)

    predictions = []
    for idx, sample in enumerate(raw_data):
        votes = sample_votes[idx]
        s_list = sample_scores_list[idx]
        
        best_verdict = max(votes, key=votes.get)
        
        predictions.append({
            "query_id": idx,
            "Claim": sample["claim"],
            "Label": sample.get("label", ""),
            "Verdict_BoN": best_verdict.capitalize(),
            "BoN_Verdict_list": sample["Verdict_list"],
            "scores": s_list,
            "votes": votes
        })
        
    return predictions

def calculate_recall_at_k(preds, k):
    recalls = []
    for p in preds:
        label = p["Label"].lower()
        verdicts = [v.lower() for v in p["BoN_Verdict_list"]]
        scores = p["scores"]
        # Use argsort to rank scores. Since score is 1 or 0, ties exist.
        ranked_indices = np.argsort(scores)[::-1]
        top_k_indices = ranked_indices[:k]
        total_relevant = sum(1 for v in verdicts if v == label)
        if total_relevant == 0: continue
        retrieved_relevant = sum(1 for i in top_k_indices if verdicts[i] == label)
        recalls.append(retrieved_relevant / total_relevant)
    return np.mean(recalls) if recalls else 0.0

# --- Timed Input Logic ---
def find_latest_checkpoint(output_dir, experiment_name):
    import glob
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

print("--- LoRA Checkpoint Selection ---")
user_input_val = [None]
def get_input():
    user_input_val[0] = input("Enter path to LoRA checkpoint folder (or press Enter for latest): ").strip()

input_thread = threading.Thread(target=get_input)
input_thread.daemon = True
input_thread.start()
input_thread.join(timeout=5)

# Detect the latest checkpoint automatically
latest_checkpoint = find_latest_checkpoint(config["output_dir"], config["experiment_name"])
if latest_checkpoint:
    print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
else:
    print(f"Warning: No checkpoints found in {config['output_dir']}")
    latest_checkpoint = os.path.join(config["output_dir"], config["experiment_name"]) # Final fallback

if input_thread.is_alive():
    print(f"\nTimeout (20s) reached! Proceeding with: {latest_checkpoint}")
    chosen_adapter = latest_checkpoint
else:
    chosen_adapter = user_input_val[0] if user_input_val[0] else latest_checkpoint
    print(f"Using checkpoint: {chosen_adapter}")

# --- Evaluation Execution ---
print(f"Loading base model: {config['model_name']} and adapter: {chosen_adapter}")
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model into memory...")
base_model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.environ.get("HF_TOKEN")
)

if chosen_adapter and os.path.exists(chosen_adapter):
    print(f"Applying LoRA adapters from {chosen_adapter}")
    model = PeftModel.from_pretrained(base_model, chosen_adapter)
else:
    print("Warning: No adapter path found. Using base model.")
    model = base_model

model.to(device)
evaluator = ProbabilisticEvaluator(model, tokenizer, device, use_dynamic_padding=config["use_dynamic_padding"])

print("Loading test data...")
with open(config["test_path"], "r") as f:
    test_raw = json.load(f)

if config["is_sanity"]:
    print("Sanity mode: using first 10 samples for eval.")
    test_raw = test_raw[:10]

test_predictions = run_full_evaluation(evaluator, tokenizer, test_raw, batch_size=config["batch_size"])

# Save results with random number
rand_num = random.randint(1000, 9999)
#output_filename = f"clef_predictions_test_{rand_num}_unsloth_qwen306_{chosen_adapter}.json"
chosen_adapter_name = chosen_adapter
if "/" in chosen_adapter_name:
  chosen_adapter_name = chosen_adapter_name.split("/")[-1]
output_path = f"/content/clef_predictions_val_{rand_num}_unsloth_qwen306_{chosen_adapter_name}_.json"

with open(output_path, "w") as f:
    json.dump(test_predictions, f, indent=4)
print(f"Evaluation complete. Results saved to: {output_path}")

# --- Metrics ---
y_true = [p["Label"].lower() for p in test_predictions]
y_pred = [p["Verdict_BoN"].lower() for p in test_predictions]
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

for k in [1, 3, 5]:
    r_k = calculate_recall_at_k(test_predictions, k)
    print(f"Recall@{k}: {r_k:.4f}")