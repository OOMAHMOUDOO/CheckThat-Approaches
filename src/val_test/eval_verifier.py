# ============================================================================
# eval_verifier.py
# Standalone evaluation script for the LoRA-tuned verifier.
# Assumes `train_verifier.py` has already finished training and saved
#   torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}"))
# ============================================================================

import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# ===== PATH PLACEHOLDERS — adjust as needed ================================
OUTPUT_DIR  = "/scratch/workdir/m.abdelaziz/P5_SCRATCHPAD/MASTERS/CLEF_CHECKTHAT_TASK2/TRIAL2/clef2026-checkthat-lab/task2/results/llama_3.2_3b_lora_16_pairwise_dpo_ce_lora_ft_lr_0.0001_epochs_2"

os.makedirs(OUTPUT_DIR,exist_ok=True)

BASE_MODEL  = "/scratch/workdir/m.abdelaziz/hf_local/llama_3.2_3b"
VAL_CSV     = "/scratch/workdir/m.abdelaziz/P5_SCRATCHPAD/MASTERS/CLEF_CHECKTHAT_TASK2/TRIAL2/clef2026-checkthat-lab/task2/English/clef2026_gpt4_o_mini_val.json"

# Which epoch checkpoint to load (e.g. last epoch index saved during training)
CHECKPOINT_EPOCH = 1          # 0-indexed; change to match the epoch you want

OUTPUT_CHECKPOINTS_DIR = "/scratch/workdir/m.abdelaziz/P5_SCRATCHPAD/MASTERS/CLEF_CHECKTHAT_TASK2/TRIAL2/clef2026-checkthat-lab/task2/outputs/llama_3.2_3b_lora_16_pairwise_dpo_ce_lora_ft_lr_0.0001_epochs_2"
MODEL_PATH = os.path.join(OUTPUT_CHECKPOINTS_DIR, f"model_epoch_{CHECKPOINT_EPOCH}")
MAX_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = LORA_RANK * 2
# ===========================================================================

torch.set_default_dtype(torch.float32)


# ---------- helpers --------------------------------------------------------
def remove_label_pattern(text: str) -> str:
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text.replace("\n", " ")


# ---------- model (same architecture as training) --------------------------
class CustomClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        tokenizer,
        dropout_value: float = 0.1,
        freeze_base_layer: bool = True,
        use_lora: bool = False,
        is_base_encoder: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.is_base_encoder = is_base_encoder

        self.yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id  = tokenizer.convert_tokens_to_ids("No")

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

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_token_logits = outputs.logits[:, -1, :]
        # index-0 → No, index-1 → Yes
        target_logits = last_token_logits[:, [self.no_token_id, self.yes_token_id]]
        return target_logits


# ---------- evaluator wrapper ----------------------------------------------
class VerifierEvaluator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        base_model: str,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build the *same* model architecture used during training
        self.model = CustomClassifier(
            base_model,
            tokenizer=self.tokenizer,
            use_lora=True,  
            is_base_encoder=False,
            lora_rank=LORA_RANK,
            lora_alpha=LORA_ALPHA,
        )

        # Load the saved state_dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Loaded checkpoint from {model_path}")

    # ---- batched encoding helper ------------------------------------------
    def encode_batch(self, claims, questions, verdicts, justifications, max_length=MAX_LENGTH):
        texts = []
        for c, q, v, j in zip(claims, questions, verdicts, justifications):
            text = (
                f"You are given a claim, evidences to the claim and a reasoning trace "
                f"that tries to verify the factuality of the claim using the evidences, "
                f"you are asked to verify whether the reasoning trace reached a correct "
                f"label or not, Labels are REFUTES, CONFLICTING, SUPPORT, Answer in Yes "
                f"or No, Yes if the reasoning trace is of high quality and reached a "
                f"correct final Verdict, No otherwise, "
                f"Claim: {c}\nVerdict: {v}\nJustification: {j}"
            )
            texts.append(text)

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].to(self.device),
            encoding["attention_mask"].to(self.device),
        )

    # ---- score a batch of (claim, trace) pairs → returns logits for "Yes" -----
    def score_batch(self, claims, questions, verdicts, justifications):
        ids, mask = self.encode_batch(claims, questions, verdicts, justifications)
        with torch.no_grad():
            logits = self.model(ids, mask)          # (batch, 2)
            # Return the raw "Yes" logit (index 1)
            return logits[:, 1].tolist()


# ---------- evaluation runner (reusable) -----------------------------------
SANITY_N = 10  # number of samples for the quick sanity check


def run_evaluation(evaluator, data, desc="Evaluating", batch_size=128):
    """Score every sample in `data` and return a list of prediction dicts."""
    predictions = []

    flat_inputs = []
    flat_indices = []

    for idx, sample in enumerate(data):
        for decoding_sample in range(len(sample["Reasoning_traces"])):
            justification = (
                remove_label_pattern(
                    sample["Reasoning_traces"][decoding_sample]
                ).split("Label:")[0]
            )
            flat_inputs.append({
                "claim": sample["claim"],
                "questions": sample.get("Questions", ""),
                "verdict": sample["Verdict_list"][decoding_sample].lower(),
                "justification": justification
            })
            flat_indices.append((idx, decoding_sample))

    all_scores = []
    # Batch processing
    for i in tqdm(range(0, len(flat_inputs), batch_size), desc=desc):
        batch = flat_inputs[i:i+batch_size]
        claims = [b["claim"] for b in batch]
        questions = [b["questions"] for b in batch]
        verdicts = [b["verdict"] for b in batch]
        justifications = [b["justification"] for b in batch]
        
        scores = evaluator.score_batch(claims, questions, verdicts, justifications)
        all_scores.extend(scores)

    results_by_sample = [
        {"verdict_list": [], "justification_list": [], "score_list": []} 
        for _ in range(len(data))
    ]
    
    for score, info, (sample_idx, trace_idx) in zip(all_scores, flat_inputs, flat_indices):
        orig_verdict = data[sample_idx]["Verdict_list"][trace_idx]
        results_by_sample[sample_idx]["verdict_list"].append(orig_verdict)
        results_by_sample[sample_idx]["justification_list"].append(info["justification"])
        results_by_sample[sample_idx]["score_list"].append(score)
        
    for idx, sample in enumerate(data):
        v_list = results_by_sample[idx]["verdict_list"]
        j_list = results_by_sample[idx]["justification_list"]
        s_list = results_by_sample[idx]["score_list"]
        
        best_verdict = v_list[np.argmax(np.array(s_list))]

        predictions.append({
            "query_id": idx,
            "Claim": sample["claim"],
            "Label": sample["label"],
            "Verdict_BoN": best_verdict,
            "BoN_Verdict_list": v_list,
            "Reasoning_traces": j_list,
            "score_list": s_list,
        })

    return predictions


def save_predictions(predictions, save_dir, filename="clef_predictions.json"):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w") as fp:
        json.dump(predictions, fp, indent=4)
    print(f"✓ Saved {len(predictions)} predictions to {out_path}")


# ===================== MAIN =================================================
if __name__ == "__main__":

    # 1. Build evaluator (loads pretrained weights)
    evaluator = VerifierEvaluator(
        model_path=MODEL_PATH,
        tokenizer_path=BASE_MODEL,
        base_model=BASE_MODEL,
        device="cuda",
    )

    # 2. Load test / validation data
    with open(VAL_CSV, "r") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} samples from {VAL_CSV}")

    # ── 3a. Sanity check on a small subset ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SANITY CHECK  –  first {SANITY_N} samples")
    print(f"{'='*60}")

    sanity_data = test_data[:SANITY_N]
    sanity_preds = run_evaluation(evaluator, sanity_data, desc="Sanity check")
    save_predictions(sanity_preds, os.path.join(OUTPUT_DIR, "sanity"))

    # Quick peek at sanity results
    correct = sum(1 for p in sanity_preds if p["Verdict_BoN"].lower() == p["Label"].lower())
    print(f"  Sanity accuracy: {correct}/{len(sanity_preds)}")
    print(f"{'='*60}\n")

    # ── 3b. Full evaluation ─────────────────────────────────────────────────
    print("Running full evaluation …")
    full_preds = run_evaluation(evaluator, test_data, desc="Full eval")
    save_predictions(full_preds, OUTPUT_DIR)
