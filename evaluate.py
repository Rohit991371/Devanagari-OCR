"""
evaluate.py
────────────
Evaluates the fine-tuned Donut model on the test split.

Metrics computed:
  - CER (Character Error Rate) per field
  - JSON exact match accuracy (full record)
  - Field-level F1 (partial match)
  - Confusion matrix for common substitution errors

Run from CustomOCR/ root:
  python evaluate.py

Prerequisites:
  - model_final/ (from train.py)
  - processed/test.jsonl (from merge_dataset.py)
"""

import json
import os
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import DonutProcessor, VisionEncoderDecoderModel
from config import DATA_ROOT, BASE_DIR, ASSETS_DIR


# ─── CONFIG ───────────────────────────────────────────────────────────────────

# MODEL_DIR = "model_final"
# PROCESSED_DIR = "processed"
# RESULTS_DIR = "evaluation"
# TEST_FILE = os.path.join(PROCESSED_DIR, "test.jsonl")

# TASK_START_TOKEN = "<s_nepali_citizenship>"
# TASK_END_TOKEN = "</s_nepali_citizenship>"
# MAX_LENGTH = 512
# BATCH_SIZE = 1   # inference one at a time for clarity

# ─────────────────────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────────────────────

# BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = DATA_ROOT / "model_final"
PROCESSED_DIR = DATA_ROOT / "processed"
RESULTS_DIR = DATA_ROOT / "evaluation"

TEST_FILE = PROCESSED_DIR / "test.jsonl"

# Create dirs if not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DONUT TOKENS
# ─────────────────────────────────────────────────────────────

TASK_START_TOKEN = "<s_nepali_citizenship>"
TASK_END_TOKEN = "</s_nepali_citizenship>"

MAX_LENGTH = 512
BATCH_SIZE = 1

# ─────────────────────────────────────────────────────────────
# EVALUATION FIELDS
# ─────────────────────────────────────────────────────────────
# # Fields to evaluate individually
EVAL_FIELDS = [
    "name.dev",
    "name.eng",

    "gender.dev",

    "citizenship_number.dev",
    "citizenship_number.eng",

    "citizenship_type",

    "dob.bs.year",
    "dob.bs.month",
    "dob.bs.day",

    "birth_address.district.dev",
    "birth_address.municipality.dev",
    "birth_address.ward",

    "current_address.district.dev",

    "parents.father.dev",

    "issue_date.bs",

    "nationality",
]
# ─── UTILITIES ────────────────────────────────────────────────────────────────


def flatten_gt(gt_dict):
    flat = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (str, int, float)):
            flat[prefix] = str(obj).strip()
    _flatten(gt_dict)
    return flat


def xml_to_flat(xml_str):
    """Parse model XML output → flat dot-notation dict."""
    result = {}
    pattern = re.compile(r"<s_([^>]+)>(.*?)</s_\1>", re.DOTALL)
    for tag, value in pattern.findall(xml_str):
        if tag == "nepali_citizenship":
            continue
        key = tag.replace("_", ".", 1) if "_" in tag else tag
        result[key] = value.strip()
    return result


def cer(pred, truth):
    """
    Character Error Rate = edit_distance(pred, truth) / len(truth)
    Returns 0.0 for perfect match, 1.0+ for completely wrong.
    Empty truth → returns 0 if pred also empty, else 1.
    """
    if not truth:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0

    # Levenshtein distance
    pred_chars = list(pred)
    truth_chars = list(truth)
    m, n = len(pred_chars), len(truth_chars)

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if pred_chars[i-1] == truth_chars[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])

    return dp[n] / n


def exact_match(pred_flat, truth_flat, fields=None):
    """Return True if all specified fields match exactly."""
    check = fields if fields else list(truth_flat.keys())
    for field in check:
        if pred_flat.get(field, "").strip() != truth_flat.get(field, "").strip():
            return False
    return True

# ─── INFERENCE ────────────────────────────────────────────────────────────────


def run_inference(model, processor, image_path, device):
    """Run model on one image, return raw XML string."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return ""

    pixel_values = processor(
        image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = processor.tokenizer(
        TASK_START_TOKEN,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=MAX_LENGTH,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.convert_tokens_to_ids(
                TASK_END_TOKEN),
            num_beams=1,   # greedy for speed; use 4 for better accuracy
        )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

# ─── EVALUATION ───────────────────────────────────────────────────────────────


def evaluate(model, processor, device):
    """Run full evaluation on test set."""

    records = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"  Evaluating {len(records)} test samples...")

    # Accumulators
    field_cer = defaultdict(list)   # field → list of CER values
    field_exact = defaultdict(list)   # field → list of bool
    full_exact = []                  # per-record full exact match
    errors = []                  # failed inference

    per_source_exact = defaultdict(list)  # source → list of bool

    all_results = []

    for rec in tqdm(records, desc="  Evaluating"):
        image_path = rec["image_path"]
        source = rec.get("source", "unknown")

        if not os.path.exists(image_path):
            errors.append(image_path)
            continue

        # Ground truth
        gt_raw = rec["ground_truth"]
        try:
            gt_dict = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        except Exception:
            errors.append(image_path)
            continue

        truth_flat = flatten_gt(gt_dict)

        # Inference
        raw_output = run_inference(model, processor, image_path, device)
        pred_flat = xml_to_flat(raw_output)

        # Compute per-field metrics
        record_exact = True
        field_results = {}

        for field in EVAL_FIELDS:
            pred_val = pred_flat.get(field, "")
            truth_val = truth_flat.get(field, "")

            field_cer_val = cer(pred_val, truth_val)
            field_exact_val = (pred_val.strip() == truth_val.strip())

            field_cer[field].append(field_cer_val)
            field_exact[field].append(field_exact_val)
            field_results[field] = {
                "pred":    pred_val,
                "truth":   truth_val,
                "cer":     field_cer_val,
                "exact":   field_exact_val,
            }

            if not field_exact_val:
                record_exact = False

        full_exact.append(record_exact)
        per_source_exact[source].append(record_exact)

        all_results.append({
            "image_path":   image_path,
            "source":       source,
            "doc_type":     rec.get("doc_type", "unknown"),
            "full_exact":   record_exact,
            "fields":       field_results,
            "raw_output":   raw_output[:500],
        })

    return all_results, field_cer, field_exact, full_exact, per_source_exact, errors


def print_report(field_cer, field_exact, full_exact, per_source_exact, errors):
    """Print formatted evaluation report."""

    print(f"\n{'─'*65}")
    print(f"  EVALUATION REPORT")
    print(f"{'─'*65}")

    # Full exact match
    total = len(full_exact)
    n_exact = sum(full_exact)
    print(
        f"\n  Full Record Exact Match: {n_exact}/{total} ({n_exact/max(1, total)*100:.1f}%)")

    # Per source
    print(f"\n  Exact Match by Source:")
    for source, vals in sorted(per_source_exact.items()):
        n = sum(vals)
        t = len(vals)
        print(f"    {source:<22}: {n}/{t} ({n/max(1, t)*100:.1f}%)")

    # Per field CER + exact
    print(f"\n  Per-Field Metrics:")
    print(f"  {'Field':<40} {'Avg CER':>8} {'Exact%':>8} {'N':>5}")
    print(f"  {'─'*40} {'─'*8} {'─'*8} {'─'*5}")
    for field in EVAL_FIELDS:
        cers = field_cer[field]
        exacts = field_exact[field]
        if not cers:
            continue
        avg_cer = np.mean(cers)
        exact_pct = np.mean(exacts) * 100
        n = len(cers)
        flag = "⚠️ " if avg_cer > 0.2 else "✅ " if avg_cer < 0.05 else "   "
        print(f"  {flag}{field:<38} {avg_cer:>8.3f} {exact_pct:>7.1f}% {n:>5}")

    if errors:
        print(f"\n  Errors (failed inference): {len(errors)}")

    print(f"\n{'─'*65}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\nEvaluation — Nepali Citizenship OCR")
    print("─" * 60)

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model not found at {MODEL_DIR}. Run train.py first.")
        return

    if not os.path.exists(TEST_FILE):
        print(f"ERROR: {TEST_FILE} not found. Run merge_dataset.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print("  Loading model...")
    processor = DonutProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    all_results, field_cer, field_exact, full_exact, per_source_exact, errors = evaluate(
        model, processor, device
    )

    print_report(field_cer, field_exact, full_exact, per_source_exact, errors)

    # Save detailed results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results saved: {results_path}")
    print("  Next: python api.py")


if __name__ == "__main__":
    main()
