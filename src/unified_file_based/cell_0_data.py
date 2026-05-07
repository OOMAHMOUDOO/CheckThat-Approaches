import json
import random
import pandas as pd
from tqdm.auto import tqdm
import os

# --- Configuration ---
RAW_TRAIN_PATH = "/workspace/CheckThat-Task2/deepseek_audit_results_20k.jsonl"
RAW_VAL_PATH = "/workspace/CheckThat-Task2/gpt4_check_data_english.jsonl"

PROCESSED_TRAIN_PATH = "/workspace/CheckThat-Task2/train_sft_processed.json"
PROCESSED_VAL_PATH = "/workspace/CheckThat-Task2/val_sft_processed.json"

def fix_response(item):
    """
    Applies the reasoning trace fixes and replaces placeholder tags with standard <think> tags.
    """
    response = item.get("response", "")
    label = item.get("original_label", "")
    is_correct_label = item.get("is_correct_label", False)
    
    # Core logic provided by user for fixing reasoning traces
    if "<TheThink>" in response and "</TheThink>" not in response:
        completion = response + f"</TheThink>\nTherefore the verdict is wrong and should be {label}."
        completion = completion.replace("<TheThink>", "<think>").replace("</TheThink>", "</think>")
    elif "</TheThink>" in response:
        parts = response.split("</TheThink>")
        first_part = parts[0] + "</TheThink>"
        # If is_correct_label is False or "false" string, it's a mismatch
        if is_correct_label is False or str(is_correct_label).lower() == "false":
            completion = first_part + f"\nTherefore the verdict is wrong and should be {label}."
        else:
            completion = first_part + "\nYes, verdict is correct."
        completion = completion.replace("<TheThink>", "<think>").replace("</TheThink>", "</think>")
    else:
        completion = response.replace("<TheThink>", "<think>").replace("</TheThink>", "</think>")
    
    return {
        "query_id": item.get("query_id"),
        "input_text": item.get("input_text"),
        "response": completion,
        "original_label": label,
        "original_verdict": item.get("original_verdict"),
        "is_correct_label": is_correct_label
    }

def stratify_data(data, name="Data"):
    """
    Balances the dataset between matching and mismatching verdicts.
    """
    matches = [item for item in data if str(item["original_verdict"]).lower() == str(item["original_label"]).lower()]
    mismatches = [item for item in data if str(item["original_verdict"]).lower() != str(item["original_label"]).lower()]
    
    min_len = min(len(matches), len(mismatches))
    print(f"[{name}] Stratifying: {len(matches)} matches, {len(mismatches)} mismatches. Balancing to {min_len} each.")
    
    random.shuffle(matches)
    random.shuffle(mismatches)
    
    balanced = matches[:min_len] + mismatches[:min_len]
    random.shuffle(balanced)
    return balanced

def main():
    print("Loading raw data...")
    try:
        with open(RAW_TRAIN_PATH, "r") as f: train_raw = json.load(f)
        with open(RAW_VAL_PATH, "r") as f: val_raw = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find raw data files. {e}")
        return

    print("Fixing responses...")
    train_fixed = [fix_response(item) for item in tqdm(train_raw, desc="Processing Train")]
    val_fixed = [fix_response(item) for item in tqdm(val_raw, desc="Processing Val")]

    print("Stratifying data...")
    train_final = stratify_data(train_fixed, name="Train")
    val_final = stratify_data(val_fixed, name="Val")

    print(f"Saving processed data to {PROCESSED_TRAIN_PATH} and {PROCESSED_VAL_PATH}...")
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    with open(PROCESSED_TRAIN_PATH, "w") as f: json.dump(train_final, f, indent=4)
    with open(PROCESSED_VAL_PATH, "w") as f: json.dump(val_final, f, indent=4)
    print("Data processing complete!")

if __name__ == "__main__":
    main()
