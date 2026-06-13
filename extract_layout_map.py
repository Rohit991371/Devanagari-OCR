"""
extract_layout_map.py
─────────────────────
Reads Label Studio annotation JSONs for Citizenship_new and Citizenship_old,
computes median bbox position per field across all annotated documents,
and writes one layout map JSON per document type.

Output:
  CustomOCR/Data/layout_map_citizenship_new.json
  CustomOCR/Data/layout_map_citizenship_old.json

Run from CustomOCR/ root:
  python extract_layout_map.py
"""

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from config import DATA_ROOT, BASE_DIR, ASSETS_DIR

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# SOURCES = [
#     {
#         "name": "citizenship_new",
#         "annotation_file": "Citizenship_new/Annotations/new_doc_annotation.json",
#         "output_file": "layout/layout_map_citizenship_new.json",
#     },
#     {
#         "name": "citizenship_old",
#         "annotation_file": "Citizenship_old/Annotations/old_doc_annotation.json",
#         "output_file": "layout/layout_map_citizenship_old.json",
#     },
# ]

# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = BASE_DIR / "Data"
LAYOUT_DIR = DATA_ROOT / "Storage" / "layout"
LAYOUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SOURCES = [
    {
        "name": "citizenship_new",
        "annotation_file": DATA_ROOT / "Storage" / "Citizenship_new" / "Annotations" / "new_doc_annotation.json",
        "output_file": LAYOUT_DIR / "layout_map_citizenship_new.json",
    },
    {
        "name": "citizenship_old",
        "annotation_file": DATA_ROOT / "Storage" / "Citizenship_old" / "Annotations" / "old_doc_annotation.json",
        "output_file": LAYOUT_DIR / "layout_map_citizenship_old.json",
    },
]


# ─── CORE LOGIC ───────────────────────────────────────────────────────────────

def extract_field_bboxes(annotation_json_path: str) -> dict:
    """
    Parse Label Studio JSON array.
    Returns: { field_label: [ {x, y, width, height}, ... ] }
    One entry per document that has that field annotated.
    """
    with open(annotation_json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)  # top-level array

    # field_label → list of bbox dicts (one per document)
    field_bboxes = defaultdict(list)

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            print(
                f"  ⚠️  Task id={task.get('id')} has no annotations — skipping")
            continue

        # Use the first (latest) annotation
        result_items = annotations[0].get("result", [])

        # Group result items by shared `id` → links rectanglelabels ↔ textarea
        by_id = defaultdict(dict)
        for item in result_items:
            item_id = item["id"]
            item_type = item["type"]
            if item_type == "rectanglelabels":
                labels = item["value"].get("rectanglelabels", [])
                if labels:
                    by_id[item_id]["label"] = labels[0]
                    by_id[item_id]["x"] = item["value"]["x"]
                    by_id[item_id]["y"] = item["value"]["y"]
                    by_id[item_id]["width"] = item["value"]["width"]
                    by_id[item_id]["height"] = item["value"]["height"]
                    by_id[item_id]["original_width"] = item.get(
                        "original_width", 1240)
                    by_id[item_id]["original_height"] = item.get(
                        "original_height", 1755)
            elif item_type == "textarea":
                texts = item["value"].get("text", [])
                by_id[item_id]["text"] = texts[0] if texts else ""

        # Collect bboxes per field label
        for item_id, data in by_id.items():
            label = data.get("label")
            if not label:
                continue  # textarea without matching rectanglelabels
            field_bboxes[label].append({
                "x": data["x"],
                "y": data["y"],
                "width": data["width"],
                "height": data["height"],
                "original_width": data.get("original_width", 1240),
                "original_height": data.get("original_height", 1755),
                "sample_text": data.get("text", ""),
            })

    return field_bboxes


def compute_layout_map(field_bboxes: dict) -> dict:
    """
    For each field, compute median x, y, width, height across all documents.
    Also records: count, sample_texts (up to 3), std_dev for QA.
    """
    layout_map = {}

    for field_label, bbox_list in sorted(field_bboxes.items()):
        xs = [b["x"] for b in bbox_list]
        ys = [b["y"] for b in bbox_list]
        widths = [b["width"] for b in bbox_list]
        heights = [b["height"] for b in bbox_list]
        ow = [b["original_width"] for b in bbox_list]
        oh = [b["original_height"] for b in bbox_list]

        layout_map[field_label] = {
            # Core placement values (used during synthesis)
            "x":      round(statistics.median(xs),      4),
            "y":      round(statistics.median(ys),      4),
            "width":  round(statistics.median(widths),  4),
            "height": round(statistics.median(heights), 4),

            # Original image dimensions (median)
            "original_width":  round(statistics.median(ow)),
            "original_height": round(statistics.median(oh)),

            # QA metadata
            "annotation_count": len(bbox_list),
            "x_std":      round(statistics.stdev(xs),      4) if len(xs) > 1 else 0.0,
            "y_std":      round(statistics.stdev(ys),      4) if len(ys) > 1 else 0.0,
            "sample_texts": list({b["sample_text"] for b in bbox_list if b["sample_text"]})[:3],
        }

    return layout_map


def print_layout_summary(layout_map: dict, source_name: str):
    print(f"\n{'─'*60}")
    print(f"  Layout Map — {source_name}")
    print(f"{'─'*60}")
    print(f"  {'FIELD':<35} {'X':>6} {'Y':>6} {'W':>6} {'H':>6}  {'N':>3}  SAMPLE")
    print(f"  {'─'*35} {'─'*6} {'─'*6} {'─'*6} {'─'*6}  {'─'*3}  {'─'*20}")
    for field, vals in sorted(layout_map.items()):
        sample = (vals["sample_texts"][0][:20]
                  if vals["sample_texts"] else "—")
        print(
            f"  {field:<35} "
            f"{vals['x']:>6.2f} "
            f"{vals['y']:>6.2f} "
            f"{vals['width']:>6.2f} "
            f"{vals['height']:>6.2f}  "
            f"{vals['annotation_count']:>3}  "
            f"{sample}"
        )
    print(f"\n  Total fields: {len(layout_map)}")


def flag_high_variance(layout_map: dict, threshold: float = 3.0):
    """Warn if any field has high std_dev — indicates inconsistent annotation."""
    warnings = []
    for field, vals in layout_map.items():
        if vals["x_std"] > threshold or vals["y_std"] > threshold:
            warnings.append(
                f"  ⚠️  High variance — {field}: "
                f"x_std={vals['x_std']:.2f}, y_std={vals['y_std']:.2f}"
            )
    if warnings:
        print(f"\n  Variance Warnings (std > {threshold}%):")
        for w in warnings:
            print(w)
    else:
        print(f"\n  ✅ All fields have low bbox variance (std ≤ {threshold}%)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n🗺️  Extracting Layout Maps from Label Studio Annotations\n")

    for source in SOURCES:
        name = source["name"]
        annotation_path = source["annotation_file"]
        output_path = source["output_file"]

        # Skip if annotation file doesn't exist yet
        if not os.path.exists(annotation_path):
            print(
                f"⏭️  Skipping {name} — annotation file not found: {annotation_path}")
            continue

        print(f"📂 Processing: {annotation_path}")

        # Parse
        field_bboxes = extract_field_bboxes(annotation_path)
        print(
            f"   Found {len(field_bboxes)} unique fields across all documents")

        # Compute median layout
        layout_map = compute_layout_map(field_bboxes)

        # Print summary table
        print_layout_summary(layout_map, name)

        # Flag high-variance fields
        flag_high_variance(layout_map)

        # Save output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(layout_map, f, ensure_ascii=False, indent=2)

        print(f"\n  💾 Saved → {output_path}")

    print("\n\n✅ Done. Layout maps ready for synthetic generation.\n")


if __name__ == "__main__":
    main()
