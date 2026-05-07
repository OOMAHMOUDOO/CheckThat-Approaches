"""
Cell 4: Generative Thinking Evaluation
Generates real model thinking, then scores P(verdict_suffix | thinking)
using suffix log-likelihoods. 3 candidates per trace (1 Yes + 2 No),
softmax over the 3 → p(Yes) is the trace score.
"""

import os
import json
import re
import random
import threading
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict

# --- Configuration ---
config = {
    "test_path": "/workspace/CheckThat-Task2/clef_2026_final_english_test.json",
    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "experiment_name": "meta-3b-sft-clef-20k",
    "output_dir": "/workspace/qwen3-4b-sft",
    "gen_batch_size": 64,       # batch size for thinking generation
    "score_batch_size": 256,    # batch size for suffix scoring (no generation, can be larger)
    "max_gen_tokens": 1024,    # max new tokens for thinking generation
    "max_seq_length": 2048,
    "is_sanity": False,
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
    "### Retrieved Evidence:\n{evidence}\n\n"
    "### Previous Verdict:\n{verdict}\n\n"
    "### Previous Justification:\n{justification}\n\n"
    "Audit this verdict and justification according to the criteria and definitions. "
    "Provide your reasoning in <think> tags, then output your final assessment in the exact required format."
)

ALL_VERDICTS = ["true", "false", "conflicting"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Utilities ---

def remove_label_pattern(text):
    """Clean up reasoning traces by removing label artifacts."""
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "", text, flags=re.IGNORECASE
    ).strip()
    return text.replace("\n", " ")


def extract_thinking(text):
    """Extract everything up to and including </think>."""
    idx = text.find("</think>")
    if idx != -1:
        return text[:idx + len("</think>")]
    return text  # didn't hit </think>, use all generated tokens


def get_verdict_suffixes(original_verdict):
    """
    Returns 3 (suffix_text, label) tuples: 1 Yes + 2 No.
    The 'Yes' maps to the original verdict label.
    The 2 'No' map to the other two verdict labels.
    """
    v = original_verdict.lower()
    alt_verdicts = [av for av in ALL_VERDICTS if av != v]
    return [
        ("\nYes, verdict is correct.", v),
        (f"\nNo, verdict should be {alt_verdicts[0].capitalize()}.", alt_verdicts[0]),
        (f"\nNo, verdict should be {alt_verdicts[1].capitalize()}.", alt_verdicts[1]),
    ]


# =============================================
# Phase 1: Batch Generate Thinking
# =============================================

def batch_generate_thinking(model, tokenizer, prompts, max_new_tokens=1024, batch_size=8):
    """Generate thinking traces for all prompts in batches (greedy, stop at </think>)."""
    all_thinkings = []

    # Left-pad for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Thinking"):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=config["max_seq_length"]
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stop_strings=["</think>"],
                tokenizer=tokenizer,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for j in range(len(batch)):
            generated_ids = outputs[j][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            thinking = extract_thinking(generated_text)
            all_thinkings.append(thinking)
            
            # Debug prints for slowness
            if j == 0: # Print only first item in batch to avoid massive spam
                status_color = "\033[32m" if "</think>" in generated_text else "\033[31m"
                print(f"\n  \033[34m[Batch {i//batch_size}]\033[0m First Item Stats:")
                print(f"    - Tokens Generated: \033[33m{len(generated_ids)}\033[0m")
                print(f"    - Thinking Chars: {len(thinking)}")
                print(f"    - Hit </think>? {status_color}{'Yes' if '</think>' in generated_text else 'NO (HIT MAX TOKENS)'}\033[0m")
                print(f"    - Sample Thinking: \033[90m{thinking[:150].replace(chr(10), ' ')}...\033[0m")

            # Log ~10% of thinkings for inspection
            if random.random() < 0.1:
                with open("thinking_samples.log", "a", encoding="utf-8") as log_f:
                    log_f.write(f"\n{'='*60}\n")
                    log_f.write(f"[Batch {i//batch_size}, Item {j}] Prompt (last 200 chars):\n")
                    log_f.write(f"...{batch[j][-200:]}\n")
                    log_f.write(f"--- Generated Thinking ---\n")
                    log_f.write(thinking + "\n")

        # Free GPU memory between batches
        del inputs, outputs
        torch.cuda.empty_cache()

    tokenizer.padding_side = original_padding_side
    return all_thinkings


# =============================================
# Phase 2: Score Suffixes (log-likelihood)
# =============================================

def score_suffixes_batch(model, tokenizer, batch_prefixes, batch_suffixes):
    """
    Compute log-likelihood of each suffix given its prefix.
    Score = sum of log-probs over suffix tokens only.
    """
    texts = [p + s for p, s in zip(batch_prefixes, batch_suffixes)]

    encoding = tokenizer(
        texts, truncation=True, padding=True,
        max_length=4096, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get prefix lengths (unpadded) to know where suffix starts
    prefix_encoding = tokenizer(batch_prefixes, padding=False, truncation=True, max_length=4096)
    prefix_lengths = [len(ids) for ids in prefix_encoding["input_ids"]]

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask=attention_mask).logits

    # Shift for next-token prediction alignment
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    scores = []
    for i in range(len(batch_prefixes)):
        p_len = prefix_lengths[i]
        valid_len = int(shift_mask[i].sum().item())

        if valid_len <= p_len - 1:
            scores.append(-9999.0)
            continue

        # Sum log-likelihood over suffix tokens only
        suffix_loss = token_losses[i, p_len - 1:valid_len].sum().item()
        scores.append(-suffix_loss)  # negative loss = log-likelihood

    return scores


# =============================================
# Main Evaluation Pipeline
# =============================================

def run_evaluation(model, tokenizer, raw_data, config):
    gen_batch_size = config["gen_batch_size"]
    score_batch_size = config["score_batch_size"]

    # ----- Step 1: Build all prompts -----
    all_prompts = []
    all_meta = []

    for idx, sample in enumerate(tqdm(raw_data, desc="Building Prompts")):
        claim = sample.get("claim", "")
        evidence = " ".join(sample.get("evidences",  []))

        for t_idx, trace in enumerate(sample["Reasoning_traces"]):
            justification = remove_label_pattern(trace).split("Label:")[0].strip()
            v = sample["Verdict_list"][t_idx]

            user_input = USER_TEMPLATE.format(
                claim=claim,
                evidence=evidence,
                verdict=v.capitalize(),
                justification=justification
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            all_prompts.append(prompt_text)
            all_meta.append({
                "sample_idx": idx,
                "trace_idx": t_idx,
                "verdict": v.lower(),
            })

    print(f"Total (sample, trace) pairs: {len(all_prompts)}")

    # ----- Step 2: Batch generate thinking -----
    print("\n--- Phase 1: Generating Thinking Traces ---")
    all_thinkings = batch_generate_thinking(
        model, tokenizer, all_prompts,
        max_new_tokens=config["max_gen_tokens"],
        batch_size=gen_batch_size
    )

    # ----- Step 3: Build scoring sequences -----
    print("\n--- Phase 2: Building Scoring Sequences ---")
    flat_prefixes = []
    flat_suffixes = []
    flat_score_meta = []

    for i, (prompt, thinking, meta) in enumerate(zip(all_prompts, all_thinkings, all_meta)):
        prefix = prompt + thinking  # full prompt + generated thinking (up to </think>)
        suffixes = get_verdict_suffixes(meta["verdict"])

        for suffix_text, suffix_label in suffixes:
            flat_prefixes.append(prefix)
            flat_suffixes.append(suffix_text)
            flat_score_meta.append({
                "pair_idx": i,  # index into all_prompts/all_meta
                "suffix_label": suffix_label,
                "is_yes": suffix_text.startswith("\nYes"),
            })

    print(f"Total scoring sequences: {len(flat_prefixes)} ({len(all_prompts)} pairs × 3 suffixes)")

    # ----- Step 4: Batch score suffixes -----
    print("\n--- Phase 3: Scoring Suffixes ---")
    tokenizer.padding_side = "right"  # right-pad for scoring
    all_scores = []

    for i in tqdm(range(0, len(flat_prefixes), score_batch_size), desc="Scoring Batches"):
        b_pref = flat_prefixes[i:i + score_batch_size]
        b_suff = flat_suffixes[i:i + score_batch_size]
        scores = score_suffixes_batch(model, tokenizer, b_pref, b_suff)
        all_scores.extend(scores)

        # Free memory
        torch.cuda.empty_cache()

    # ----- Step 5: Aggregate scores -----
    print("\n--- Phase 4: Aggregating Results ---")

    # Group scores by (pair_idx) → each pair has exactly 3 scores
    pair_scores = defaultdict(list)
    for score, smeta in zip(all_scores, flat_score_meta):
        pair_scores[smeta["pair_idx"]].append({
            "label": smeta["suffix_label"],
            "score": score,
            "is_yes": smeta["is_yes"],
        })

    # Accumulate votes per sample
    sample_votes = [{"true": 0.0, "false": 0.0, "conflicting": 0.0} for _ in range(len(raw_data))]
    sample_yes_scores = [[] for _ in range(len(raw_data))]
    sample_thinkings = [[] for _ in range(len(raw_data))]

    for pair_idx, scores_list in pair_scores.items():
        meta = all_meta[pair_idx]
        s_idx = meta["sample_idx"]

        # Softmax over the 3 candidates
        scores_tensor = torch.tensor([item["score"] for item in scores_list])
        probs = torch.softmax(scores_tensor, dim=0).tolist()

        yes_prob = None
        for i, item in enumerate(scores_list):
            label = item["label"]
            prob = probs[i]
            sample_votes[s_idx][label] += prob
            if item["is_yes"]:
                yes_prob = prob

        sample_yes_scores[s_idx].append(yes_prob if yes_prob is not None else 0.0)
        sample_thinkings[s_idx].append(all_thinkings[pair_idx])

    # ----- Step 6: Build final predictions -----
    predictions = []
    for idx, sample in enumerate(raw_data):
        votes = sample_votes[idx]
        best_verdict = max(votes, key=votes.get)

        predictions.append({
            "query_id": idx,
            "Claim": sample["claim"],
            "Label": sample.get("label", ""),
            "Verdict_BoN": best_verdict.capitalize(),
            "BoN_Verdict_list": sample["Verdict_list"],
            "score_list": sample_yes_scores[idx],
            "votes": votes,
            "generated_thinkings": sample_thinkings[idx],
        })

    return predictions


# =============================================
# Checkpoint Selection (reused from cell_3)
# =============================================

def find_latest_checkpoint(output_dir, experiment_name):
    import glob
    pattern = os.path.join(output_dir, f"{experiment_name}*_epoch_*")
    folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]

    if not folders:
        fallback = os.path.join(output_dir, experiment_name)
        return fallback if os.path.exists(fallback) else None

    def sort_key(folder_path):
        folder_name = os.path.basename(folder_path)
        match = re.search(r"epoch_(\d+)", folder_name)
        epoch_num = int(match.group(1)) if match else -1
        return (epoch_num, folder_name)

    folders.sort(key=sort_key, reverse=True)
    return folders[0]


# =============================================
# Execution
# =============================================

if __name__ == "__main__" or True:  # Always run (Colab compat)

    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged into HuggingFace")

    # --- Checkpoint selection ---
    print("--- LoRA Checkpoint Selection ---")
    user_input_val = [None]
    def get_input():
        user_input_val[0] = input("Enter path to LoRA checkpoint folder (or press Enter for latest): ").strip()

    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True
    input_thread.start()
    input_thread.join(timeout=20)

    latest_checkpoint = find_latest_checkpoint(config["output_dir"], config["experiment_name"])
    if latest_checkpoint:
        print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
    else:
        print(f"Warning: No checkpoints found in {config['output_dir']}")
        latest_checkpoint = os.path.join(config["output_dir"], config["experiment_name"])

    if input_thread.is_alive():
        print(f"\nTimeout reached! Proceeding with: {latest_checkpoint}")
        chosen_adapter = latest_checkpoint
    else:
        chosen_adapter = user_input_val[0] if user_input_val[0] else latest_checkpoint
        print(f"Using checkpoint: {chosen_adapter}")

    # --- Load model ---
    print(f"Loading base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    model.eval()

    # --- Load test data ---
    print("Loading test data...")
    with open(config["test_path"], "r") as f:
        test_raw = json.load(f)

    if config["is_sanity"]:
        print("Sanity mode: using first 10 samples.")
        test_raw = test_raw[:10]

    # --- Run evaluation ---
    predictions = run_evaluation(model, tokenizer, test_raw, config)

    # --- Save results ---
    rand_num = random.randint(1000, 9999)
    chosen_adapter_name = os.path.basename(chosen_adapter) if chosen_adapter else "base"
    output_path = f"/workspace/clef_predictions_thinking_{rand_num}_{chosen_adapter_name}.json"

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"\nEvaluation complete. Results saved to: {output_path}")
    print(f"Produced {len(predictions)} predictions.")

    # --- Quick accuracy if labels available ---
    labeled = [p for p in predictions if p.get("Label")]
    if labeled:
        correct = sum(1 for p in labeled if p["Verdict_BoN"].lower() == p["Label"].lower())
        print(f"Accuracy: {correct}/{len(labeled)} = {correct/len(labeled):.4f}")
