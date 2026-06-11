"""
create_templates.py
────────────────────
Reads Label Studio annotation JSONs, erases annotated field bboxes from
real document images using OpenCV inpainting, and saves blank templates.

One blank template is created per real document — preserving natural
variation in paper texture, scan quality, and background color.

Output:
  CustomOCR/Data/Templates/citizenship_new/template_001.jpg ...
  CustomOCR/Data/Templates/citizenship_old/template_001.jpg ...

Run from CustomOCR/ root:
  pip install opencv-python numpy
  python create_templates.py
"""

import json
import os
import re
import cv2
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# =========================================================
# BASE DIRECTORY
# =========================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# =========================================================
# ROOT DIRECTORIES
# =========================================================

DATA_DIR = BASE_DIR / "Data"
TEMPLATES_DIR = BASE_DIR / "Templates"

# =========================================================
# DOCUMENT SOURCES
# =========================================================

SOURCES = [
    {
        "name": "citizenship_new",

        "annotation_file":
            DATA_DIR
            / "Citizenship_new"
            / "Annotations"
            / "new_doc_annotation.json",

        "docs_dir":
            DATA_DIR
            / "Citizenship_new"
            / "Docs",

        "output_dir":
            TEMPLATES_DIR
            / "citizenship_new",
    },

    {
        "name": "citizenship_old",

        "annotation_file":
            DATA_DIR
            / "Citizenship_old"
            / "Annotations"
            / "old_doc_annotation.json",

        "docs_dir":
            DATA_DIR
            / "Citizenship_old"
            / "Docs",

        "output_dir":
            TEMPLATES_DIR
            / "citizenship_old",
    },
]

# =========================================================
# CREATE OUTPUT DIRECTORIES
# =========================================================

for source in SOURCES:
    source["output_dir"].mkdir(parents=True, exist_ok=True)

# =========================================================
# DEBUG
# =========================================================

print("\nLoaded Sources")
print("=" * 50)

for source in SOURCES:
    print(f"\nSOURCE: {source['name']}")
    print("Annotation:", source["annotation_file"])
    print("Docs Dir: ", source["docs_dir"])
    print("Output:   ", source["output_dir"])
INPAINT_RADIUS = 5
BBOX_PADDING = 4

# ─── FILENAME RESOLUTION ──────────────────────────────────────────────────────


def strip_uuid_prefix(filename):
    """Remove Label Studio UUID prefix: '5f4c6fe1-somefile.jpg' -> 'somefile.jpg'"""
    return re.sub(r'^[0-9a-f]{8}-', '', filename, flags=re.IGNORECASE)


def expand_windows_suffix(filename):
    """
    Label Studio concatenates Windows duplicate-file suffix into the page number.

    On disk (Windows):     1000386739_page-0001(1).jpg
    In Label Studio JSON:  1000386739_page-00011.jpg  <- (1) became 1, no parens

    So 'page-00011' -> try 'page-0001(1)' first, then original as fallback.
    Returns candidates most-likely-first.
    """
    candidates = [filename]
    m = re.match(r'^(.*?)(\d{2,})(\.\w+)$', filename)
    if m:
        base, digits, ext = m.group(1), m.group(2), m.group(3)
        last = digits[-1]
        if last in "123456789":
            core = digits[:-1]
            candidates.insert(0, f"{base}{core}({last}){ext}")
    return candidates


def find_image(filename, docs_dir):
    """
    Locate image on disk, handling:
      1. Label Studio UUID prefix   5f4c6fe1-name.jpg  ->  name.jpg
      2. Windows duplicate suffix   page-00011.jpg     ->  page-0001(1).jpg
      3. Case-insensitive matching
    """
    stripped = strip_uuid_prefix(filename)

    all_candidates = []
    for c in expand_windows_suffix(stripped) + expand_windows_suffix(filename):
        if c not in all_candidates:
            all_candidates.append(c)

    try:
        dir_files = os.listdir(docs_dir)
    except FileNotFoundError:
        return None

    dir_lower = {f.lower(): f for f in dir_files}

    for candidate in all_candidates:
        exact = os.path.join(docs_dir, candidate)
        if os.path.exists(exact):
            return exact
        match = dir_lower.get(candidate.lower())
        if match:
            return os.path.join(docs_dir, match)

    return None

# ─── ANNOTATION PARSING ───────────────────────────────────────────────────────


def load_annotations(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_bboxes_for_task(task):
    """Extract all rectanglelabel bboxes from a single task."""
    annotations = task.get("annotations", [])
    if not annotations:
        return []

    bboxes = []
    for item in annotations[0].get("result", []):
        if item["type"] == "rectanglelabels":
            val = item["value"]
            labels = val.get("rectanglelabels", [])
            bboxes.append({
                "label":           labels[0] if labels else "unknown",
                "x":               val["x"],
                "y":               val["y"],
                "width":           val["width"],
                "height":          val["height"],
                "original_width":  item.get("original_width",  1240),
                "original_height": item.get("original_height", 1755),
            })
    return bboxes

# ─── INPAINTING ───────────────────────────────────────────────────────────────


def pct_to_pixels(bbox, img_w, img_h):
    """Convert percentage bbox -> pixel coords with padding, clamped to image."""
    x1 = max(0,         int((bbox["x"] / 100.0) * img_w) - BBOX_PADDING)
    y1 = max(0,         int((bbox["y"] / 100.0) * img_h) - BBOX_PADDING)
    x2 = min(
        img_w - 1, int(((bbox["x"] + bbox["width"]) / 100.0) * img_w) + BBOX_PADDING)
    y2 = min(
        img_h - 1, int(((bbox["y"] + bbox["height"]) / 100.0) * img_h) + BBOX_PADDING)
    return x1, y1, x2, y2


def create_blank_template(image, bboxes):
    """Build mask over all annotated fields and inpaint."""
    img_h, img_w = image.shape[:2]
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for bbox in bboxes:
        x1, y1, x2, y2 = pct_to_pixels(bbox, img_w, img_h)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)

# ─── MAIN ─────────────────────────────────────────────────────────────────────


def process_source(source):
    name = source["name"]
    annotation_file = source["annotation_file"]
    docs_dir = source["docs_dir"]
    output_dir = source["output_dir"]

    print(f"\n{'─'*60}")
    print(f"  Processing: {name}")
    print(f"{'─'*60}")

    if not os.path.exists(annotation_file):
        print(f"  Skipping - annotation file not found: {annotation_file}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tasks = load_annotations(annotation_file)
    print(f"  Found {len(tasks)} annotated documents")

    success = skip = errors = 0

    for i, task in enumerate(tasks, 1):
        # Resolve filename
        filename = task.get("file_upload", "")
        if not filename:
            filename = os.path.basename(task.get("data", {}).get("ocr", ""))
        if not filename:
            print(f"  Task {i}: cannot determine filename - skipping")
            skip += 1
            continue

        image_path = find_image(filename, docs_dir)
        if image_path is None:
            print(f"  Task {i}: image not found in Docs/ - {filename}")
            skip += 1
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  Task {i}: cv2 could not read image - {image_path}")
            errors += 1
            continue

        img_h, img_w = image.shape[:2]
        bboxes = extract_bboxes_for_task(task)
        if not bboxes:
            print(f"  Task {i}: no bboxes found - {filename}")
            skip += 1
            continue

        try:
            template = create_blank_template(image, bboxes)
        except Exception as e:
            print(f"  Task {i}: inpainting failed - {e}")
            errors += 1
            continue

        out_filename = f"template_{i:03d}.jpg"
        out_path = os.path.join(output_dir, out_filename)
        cv2.imwrite(out_path, template, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"  OK [{i:02d}/{len(tasks)}] {os.path.basename(image_path)} "
              f"-> {out_filename}  ({img_w}x{img_h}, {len(bboxes)} fields erased)")
        success += 1

    print(
        f"\n  Summary: {success} templates created, {skip} skipped, {errors} errors")
    print(f"  Output -> {output_dir}/")


def main():
    print("\nCreating Blank Templates via Inpainting\n")
    for source in SOURCES:
        process_source(source)
    print("\nDone. Blank templates ready for synthetic generation.")
    print("Next: visually inspect a few templates in Data/Templates/")
    print("Then run: python generate_synthetic.py")


if __name__ == "__main__":
    main()
