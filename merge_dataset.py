"""
merge_dataset.py
─────────────────
Merges real + synthetic metadata.jsonl files into a single
HuggingFace-compatible dataset split into train/val/test.

Strategy:
  - ALL real docs go into train (too few to waste on val/test)
  - Synthetic: 80% train / 10% val / 10% test
  - Shuffle with fixed seed for reproducibility

Output:
  CustomOCR/Data/processed/
  ├── train.jsonl      image_path + ground_truth
  ├── val.jsonl
  ├── test.jsonl
  └── stats.json       record counts + split info

Run from CustomOCR/ root:
  python merge_dataset.py
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from config import DATA_ROOT, BASE_DIR, ASSETS_DIR


# ─────────────────────────────────────────────────────────────
# BASE DIRS
# ─────────────────────────────────────────────────────────────

# BASE_DIR = Path(__file__).resolve().parent

# DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = DATA_ROOT / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42

SOURCES = [

    # ───────── REAL DATA ─────────

    {
        "label": "real_new",

        "metadata_file":
            DATA_ROOT / "Storage" / "Citizenship_new" /
            "Annotations" / "metadata.jsonl",

        "image_base":
            DATA_ROOT / "Storage" / "Citizenship_new" / "Docs",

        "split": "train_only",

        "doc_type": "citizenship_new",
    },

    {
        "label": "real_old",

        "metadata_file":
            DATA_ROOT / "Storage" / "Citizenship_old" /
            "Annotations" / "metadata.jsonl",

        "image_base":
            DATA_ROOT / "Storage" / "Citizenship_old" / "Docs",

        "split": "train_only",

        "doc_type": "citizenship_old",
    },

    # ───────── SYNTHETIC DATA ─────────

    {
        "label": "synth_new",

        "metadata_file":
            DATA_ROOT / "Storage" / "Synthetic" /
            "citizenship_new" / "metadata.jsonl",

        "image_base":
            DATA_ROOT / "Storage" / "Synthetic" /
            "citizenship_new",

        "split": "split",

        "doc_type": "citizenship_new",
    },

    {
        "label": "synth_old",

        "metadata_file":
            DATA_ROOT / "Storage" / "Synthetic" /
            "citizenship_old" / "metadata.jsonl",

        "image_base":
            DATA_ROOT / "Synthetic" /
            "citizenship_old",

        "split": "split",

        "doc_type": "citizenship_old",
    },
]


VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ─── LOADING ──────────────────────────────────────────────────────────────────


def load_metadata(source):
    """
    Load metadata.jsonl and resolve image paths to absolute paths.
    Validates that:
      1. ground_truth is valid JSON
      2. image file exists on disk
    Returns list of clean record dicts.
    """
    metadata_file = source["metadata_file"]
    image_base = source["image_base"]
    label = source["label"]
    doc_type = source["doc_type"]

    if not os.path.exists(metadata_file):
        print(f"  ⚠️  Not found: {metadata_file} — skipping {label}")
        return []

    records = []
    skipped = 0
    bad_json = 0
    bad_image = 0

    with open(metadata_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Parse metadata line
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            # Validate ground_truth is parseable JSON
            gt_raw = record.get("ground_truth", "")
            try:
                gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
            except json.JSONDecodeError:
                bad_json += 1
                continue

            # Resolve image path
            img_relative = record.get("image", "")
            img_path = os.path.join(image_base, img_relative)

            # Also try stripping leading "images/" if already in base
            if not os.path.exists(img_path):
                alt = os.path.join(image_base, os.path.basename(img_relative))
                if os.path.exists(alt):
                    img_path = alt
                else:
                    bad_image += 1
                    continue

            # Normalize ground_truth to always be a dict (not a string)
            if isinstance(gt_raw, str):
                gt_normalized = gt
            else:
                gt_normalized = gt_raw

            records.append({
                "image_path":   os.path.abspath(img_path),
                "ground_truth": gt_normalized,
                "doc_type":     doc_type,
                "source":       label,
            })

    total = len(records) + skipped + bad_json + bad_image
    print(f"  {label:20s}: {len(records):5d} valid  "
          f"({bad_image} missing images, {bad_json} bad JSON)")

    return records


# ─── SPLITTING ────────────────────────────────────────────────────────────────

def split_records(records, val_ratio, test_ratio, seed):
    """Split records into train/val/test lists."""
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


# ─── WRITING ──────────────────────────────────────────────────────────────────

def write_split(records, path):
    """Write records to jsonl. ground_truth serialized as JSON string."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            out = {
                "image_path":   rec["image_path"],
                "ground_truth": json.dumps(rec["ground_truth"], ensure_ascii=False),
                "doc_type":     rec["doc_type"],
                "source":       rec["source"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\nMerge Dataset")
    print("─" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    train_all = []
    val_all = []
    test_all = []
    stats = {"sources": {}}

    print("\nLoading sources...")
    for source in SOURCES:
        records = load_metadata(source)
        if not records:
            continue

        label = source["label"]

        if source["split"] == "train_only":
            # All real docs → train
            train_all.extend(records)
            stats["sources"][label] = {
                "total": len(records), "train": len(records), "val": 0, "test": 0
            }
        else:
            # Synthetic → split
            tr, va, te = split_records(
                records, VAL_RATIO, TEST_RATIO, RANDOM_SEED)
            train_all.extend(tr)
            val_all.extend(va)
            test_all.extend(te)
            stats["sources"][label] = {
                "total": len(records), "train": len(tr), "val": len(va), "test": len(te)
            }

    if not train_all:
        print("\nERROR: No training records found. Run generate_synthetic.py first.")
        return

    # Final shuffle of each split
    random.seed(RANDOM_SEED)
    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    # Write splits
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "val.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test.jsonl")

    write_split(train_all, train_path)
    write_split(val_all,   val_path)
    write_split(test_all,  test_path)

    # Write stats
    stats["splits"] = {
        "train": len(train_all),
        "val":   len(val_all),
        "test":  len(test_all),
        "total": len(train_all) + len(val_all) + len(test_all),
    }
    stats["config"] = {
        "val_ratio":   VAL_RATIO,
        "test_ratio":  TEST_RATIO,
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  Dataset Summary")
    print(f"{'─'*60}")
    print(f"  {'Source':<22} {'Total':>7} {'Train':>7} {'Val':>6} {'Test':>6}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")
    for lbl, s in stats["sources"].items():
        print(
            f"  {lbl:<22} {s['total']:>7} {s['train']:>7} {s['val']:>6} {s['test']:>6}")
    print(f"  {'─'*22} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")
    print(f"  {'TOTAL':<22} {stats['splits']['total']:>7} "
          f"{stats['splits']['train']:>7} {stats['splits']['val']:>6} "
          f"{stats['splits']['test']:>6}")

    print(f"\n  Output:")
    print(f"    {train_path}  ({len(train_all)} records)")
    print(f"    {val_path}    ({len(val_all)} records)")
    print(f"    {test_path}   ({len(test_all)} records)")
    print(f"\n  Next: python train.py")


if __name__ == "__main__":
    main()
