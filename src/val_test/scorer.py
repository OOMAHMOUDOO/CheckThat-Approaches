"""
Evaluate RM prediction quality on the CLEF dataset.

Metrics computed:
  1. IR-style Recall@k and Precision@k (k=1..5):
       Recall@k    = (# relevant in top-k) / (# relevant in full 15-trace list)
       Precision@k = (# relevant in top-k) / k
     A trace is relevant if its verdict matches the ground truth label.
     Per-sample values saved to per_sample_ir.csv; means saved to result.csv.
  2. Per-class precision, recall, F1 (based on Verdict_BoN vs Label).
  3. Macro and weighted precision, recall, F1.

Output:
  output/RM_prediction/result.csv          — average / aggregate metrics
  output/RM_prediction/per_sample_ir.csv   — per-sample Recall@k and Precision@k
"""

import argparse
import csv
import json
import os

from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description="Evaluate RM prediction quality.")
parser.add_argument("--input_path", type=str, default="output/RM_prediction/clef_predictions.json", help="Path to predictions JSON")
parser.add_argument("--output_dir", type=str, default="output/RM_prediction", help="Directory to save output CSVs")
args = parser.parse_args()

INPUT_PATH      = args.input_path
OUTPUT_PATH     = os.path.join(args.output_dir, "result.csv")
PER_SAMPLE_PATH = os.path.join(args.output_dir, "per_sample_ir.csv")
CLASSES         = ["false", "true", "conflicting"]
K_VALUES        = list(range(1, 6))


with open(INPUT_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)

seen: set = set()
unique_data = []
for entry in raw_data:
    if entry["Claim"] not in seen:
        seen.add(entry["Claim"])
        unique_data.append(entry)

print(f"Total unique claims: {len(unique_data)}")


# ── 1. IR-style Recall@k and Precision@k ─────────────────────────────────────
def ir_metrics_at_k(entry: dict, k: int) -> tuple[float, float]:
    """
    Compute IR-style Recall@k and Precision@k for a single claim.

    Recall@k    = (# relevant in top-k) / (# relevant in full 15-trace list)
    Precision@k = (# relevant in top-k) / k

    A trace is relevant when its verdict matches the ground truth label.
    If no trace in the full list is relevant, Recall@k is defined as 0.
    """
    label    = entry["Label"].lower()
    verdicts = [v.lower() for v in entry["BoN_Verdict_list"]]

    ranked = sorted(range(len(entry["scores"])),
                    key=lambda i: entry["scores"][i],
                    reverse=True)

    total_relevant     = sum(1 for v in verdicts if v == label)
    top_k_verdicts     = [verdicts[i] for i in ranked[:k]]
    retrieved_relevant = sum(1 for v in top_k_verdicts if v == label)

    recall_k    = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
    precision_k = retrieved_relevant / k

    return recall_k, precision_k


# Accumulate per-sample records and running sums for means
per_sample_records = []
sum_recall    = {k: 0.0 for k in K_VALUES}
sum_precision = {k: 0.0 for k in K_VALUES}

for entry in unique_data:
    record = {"query_id": entry["query_id"], "claim": entry["Claim"][:80]}
    for k in K_VALUES:
        r, p = ir_metrics_at_k(entry, k)
        record[f"recall@{k}"]    = round(r, 4)
        record[f"precision@{k}"] = round(p, 4)
        sum_recall[k]    += r
        sum_precision[k] += p
    per_sample_records.append(record)

n = len(unique_data)
mean_recall    = {k: sum_recall[k]    / n for k in K_VALUES}
mean_precision = {k: sum_precision[k] / n for k in K_VALUES}


# ── 2 & 3. Classification metrics on Verdict_BoN ─────────────────────────────
y_true = [e["Label"].lower()       for e in unique_data]
y_pred = [e["Verdict_BoN"].lower() for e in unique_data]

report = classification_report(
    y_true, y_pred,
    labels=CLASSES,
    output_dict=True,
    zero_division=0,
)


# ── Save aggregate metrics → result.csv ───────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)

    # Section 1 — IR ranking metrics
    w.writerow(["=== IR Ranking Metrics (RM-ranked reasoning traces) ==="])
    w.writerow(["k", "Mean Recall@k", "Mean Precision@k"])
    for k in K_VALUES:
        w.writerow([k, round(mean_recall[k], 4), round(mean_precision[k], 4)])
    w.writerow([])

    # Section 2 — Per-class classification metrics
    w.writerow(["=== Per-Class Metrics (Verdict_BoN vs Ground Truth) ==="])
    w.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for cls in CLASSES:
        r = report[cls]
        w.writerow([cls,
                    round(r["precision"], 4),
                    round(r["recall"],    4),
                    round(r["f1-score"],  4),
                    int(r["support"])])
    w.writerow([])

    # Section 3 — Macro & weighted aggregates
    w.writerow(["=== Aggregate Metrics ==="])
    w.writerow(["Average", "Precision", "Recall", "F1-Score"])
    for avg_type in ["macro avg", "weighted avg"]:
        r = report[avg_type]
        w.writerow([avg_type,
                    round(r["precision"], 4),
                    round(r["recall"],    4),
                    round(r["f1-score"],  4)])


# ── Save per-sample IR metrics → per_sample_ir.csv ───────────────────────────
per_sample_header = (
    ["query_id", "claim"]
    + [f"recall@{k}"    for k in K_VALUES]
    + [f"precision@{k}" for k in K_VALUES]
)

with open(PER_SAMPLE_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=per_sample_header)
    w.writeheader()
    w.writerows(per_sample_records)


# ── Print summary ─────────────────────────────────────────────────────────────
print("\n--- IR Ranking Metrics ---")
print(f"  {'k':>3}  {'Mean Recall@k':>14}  {'Mean Precision@k':>16}")
for k in K_VALUES:
    print(f"  {k:>3}  {mean_recall[k]:>14.4f}  {mean_precision[k]:>16.4f}")

print("\n--- Per-Class Metrics (Verdict_BoN) ---")
for cls in CLASSES:
    r = report[cls]
    print(f"  {cls:>12s}:  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}  n={int(r['support'])}")

print("\n--- Aggregate Metrics ---")
for avg_type in ["macro avg", "weighted avg"]:
    r = report[avg_type]
    print(f"  {avg_type}:  P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1-score']:.4f}")

print(f"\nAggregate results  -> {OUTPUT_PATH}")
print(f"Per-sample results -> {PER_SAMPLE_PATH}")
