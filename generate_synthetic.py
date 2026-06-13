"""
generate_synthetic.py  (v2 — alignment fix + handwriting overlay)
──────────────────────────────────────────────────────────────────
Generates synthetic citizenship documents:
  1. Randomly picks a blank template
  2. Generates realistic random field values
  3. Stamps text using FIXED alignment (anchor="lm", per-field font sizing)
  4. Overlays handwritten word images for select fields (if cache exists)
  5. Applies augmentation
  6. Saves image + metadata.jsonl matching real data format

Prerequisites (run first):
  python extract_layout_map.py
  python create_templates.py
  python build_handwritten_words.py   ← optional but recommended

Run from CustomOCR/ root:
  python generate_synthetic.py
"""

import json
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from tqdm import tqdm
from config import DATA_ROOT, BASE_DIR, ASSETS_DIR


# ─────────────────────────────────────────────────────────────
# BASE DIRS
# ─────────────────────────────────────────────────────────────

# BASE_DIR = Path(__file__).resolve().parent
# ASSETS_DIR = BASE_DIR / "assets"
# DATA_DIR = BASE_DIR / "Data"

FONTS_DIR = ASSETS_DIR / "fonts"

HW_INDEX_PATH = (
    DATA_ROOT /
    "handwriting_cache" /
    "index.json"
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

SOURCES = [
    {
        "name": "citizenship_new",

        "templates_dir":
            DATA_ROOT / "Storage" / "Templates" / "citizenship_new",

        "layout_map":
            DATA_ROOT / "layout" /
            "layout_map_citizenship_new.json",

        "output_dir":
            DATA_ROOT / "Synthetic" / "citizenship_new",

        "count": 3000,
        "doc_type": "new",

        # handwritten fields
        "hw_fields": [
            "officer_name"
        ],

        "hw_probability": 0.10,
    },

    {
        "name": "citizenship_old",

        "templates_dir":
            DATA_ROOT / "Storage" / "Templates" / "citizenship_old",

        "layout_map":
            DATA_ROOT / "layout" /
            "layout_map_citizenship_old.json",

        "output_dir":
            DATA_ROOT / "Synthetic" / "citizenship_old",

        "count": 2000,
        "doc_type": "old",

        # handwritten fields
        "hw_fields": [
            "name_dev",
            "father_name",
            "dob_year_bs",
            "dob_month_bs",
            "dob_day_bs",
            "issue_date",
            "citizenship_number",
        ],

        "hw_probability": 0.40,
    },
]

# ─────────────────────────────────────────────────────────────
# FONT PATHS
# ─────────────────────────────────────────────────────────────

# DEV_FONT = (
#     FONTS_DIR /
#     "Noto_Sans_Devanagari" /
#     "NotoSansDevanagari-VariableFont_wdth,wght.ttf"
# )

# ENG_FONT = (
#     FONTS_DIR /
#     "Arial.ttf"
# )


# Font size: fit to bbox height. This ratio controls how much of the bbox height the font fills.
FONT_HEIGHT_FILL = 0.72    # font size = 72% of bbox pixel height

# Augmentation probabilities
AUG_BLUR_PROB = 0.4
AUG_BRIGHTNESS_PROB = 0.5
AUG_NOISE_PROB = 0.3
AUG_ROTATION_PROB = 0.3
AUG_ROTATION_MAX = 2.5

# ─── DIGIT MAPS ───────────────────────────────────────────────────────────────

ENG_TO_DEV = str.maketrans("0123456789", "०१२३४५६७८९")
DEV_TO_ENG = str.maketrans("०१२३४५६७८९", "0123456789")


def to_dev(s): return str(s).translate(ENG_TO_DEV)
def to_eng(s): return str(s).translate(DEV_TO_ENG)


AD_MONTHS_ENG = ["JAN", "FEB", "MAR", "APR", "MAY",
                 "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
CITIZENSHIP_TYPES = ["वंशज", "जन्म", "अंगिकृत", "बैवाहिक अंगिकृत"]

# ─── ASSET LOADING ────────────────────────────────────────────────────────────

_assets_cache = {}


def load_json(path):
    if path not in _assets_cache:
        with open(path, "r", encoding="utf-8") as f:
            _assets_cache[path] = json.load(f)
    return _assets_cache[path]


def load_hw_index():
    """Load handwriting cache index. Returns empty dict if not built yet."""
    if os.path.exists(HW_INDEX_PATH):
        with open(HW_INDEX_PATH, "r", encoding="utf-8") as f:
            index = json.load(f)
        print(f"  Handwriting cache loaded: {len(index)} words")
        return index
    else:
        print(f"  ⚠️  Handwriting cache not found at {HW_INDEX_PATH}")
        print(f"  Run build_handwritten_words.py first for handwriting overlay.")
        print(f"  Continuing with TTF font only.\n")
        return {}


def load_fonts_for_bbox(bbox_h_px):
    """
    Load PIL fonts sized to a specific bbox height.
    font_size = bbox_h_px * FONT_HEIGHT_FILL
    Returns dict of {key: ImageFont}
    """
    font_size = max(10, int(bbox_h_px * FONT_HEIGHT_FILL))
    fonts = {}
    paths = {
        "dev": os.path.join(FONTS_DIR, "dev", "NotoSansDevanagari-Regular.ttf"),
        "eng": os.path.join(FONTS_DIR, "eng", "NotoSans-Regular.ttf"),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            fonts[key] = ImageFont.truetype(path, font_size)
        else:
            fonts[key] = ImageFont.load_default()
    return fonts

# ─── DATA GENERATORS ──────────────────────────────────────────────────────────


def random_bs_date(year_min=2030, year_max=2065):
    bs_year = random.randint(year_min, year_max)
    bs_month = random.randint(1, 12)
    bs_day = random.randint(1, 30)
    ad_year = bs_year - 57
    return {
        "bs": {
            "year":  to_dev(bs_year),
            "month": to_dev(str(bs_month).zfill(2)),
            "day":   to_dev(str(bs_day).zfill(2)),
        },
        "ad": {
            "year":  str(ad_year),
            "month": AD_MONTHS_ENG[bs_month - 1],
            "day":   str(bs_day).zfill(2),
        }
    }


def random_bs_issue_date(dob_bs_year_dev):
    dob_year = int(str(dob_bs_year_dev).translate(DEV_TO_ENG))
    issue_year = dob_year + random.randint(16, 40)
    issue_month = random.randint(1, 12)
    issue_day = random.randint(1, 28)
    return (f"{to_dev(issue_year)}-"
            f"{to_dev(str(issue_month).zfill(2))}-"
            f"{to_dev(str(issue_day).zfill(2))}")


def random_citizenship_number_new():
    d = random.randint(1, 77)
    p = random.randint(1, 7)
    y = random.randint(60, 82)
    seq = random.randint(1000, 99999)
    eng = f"{d:02d}-0{p}-{y:02d}-{seq:05d}"
    return {"dev": to_dev(eng), "eng": eng}


def random_citizenship_number_old():
    num = random.randint(100000000, 999999999)
    return {"dev": to_dev(str(num)), "eng": str(num)}


def random_address(districts, municipalities):
    district = random.choice(districts)
    dist_dev = district["dev"]
    if dist_dev in municipalities:
        pool = municipalities[dist_dev]["municipalities"]
        muni = random.choice(pool)
        dist_eng = municipalities[dist_dev]["eng"]
        muni_dev = muni["dev"]
        muni_eng = muni["eng"]
    else:
        dist_eng = district["eng"]
        muni_dev = dist_dev + " नगरपालिका"
        muni_eng = dist_eng + " Municipality"
    ward = random.randint(1, 20)
    return {
        "district":     {"dev": dist_dev,  "eng": dist_eng},
        "municipality": {"dev": muni_dev,  "eng": muni_eng},
        "ward":         to_dev(str(ward)),
    }


def build_gt_new(names, districts, municipalities):
    name = random.choice(names["male"] + names["female"])
    father = random.choice(names["male"])
    mother = random.choice(names["female"])
    dob = random_bs_date(2030, 2065)
    birth = random_address(districts, municipalities)
    curr = random_address(districts, municipalities)
    cn = random_citizenship_number_new()
    issue = random_bs_issue_date(dob["bs"]["year"])
    gender_dev = random.choice(["महिला", "पुरुष"])
    gender_eng = "Female" if gender_dev == "महिला" else "Male"
    return {
        "name":               {"dev": name["dev"],   "eng": name["eng"]},
        "gender":             {"dev": gender_dev,    "eng": gender_eng},
        "citizenship_number": {"dev": cn["dev"],     "eng": cn["eng"]},
        "dob":                {"bs": dob["bs"],       "ad": dob["ad"]},
        "birth_address":      birth,
        "current_address":    curr,
        "parents": {
            "father": {"dev": father["dev"], "eng": father["eng"]},
            "mother": {"dev": mother["dev"], "eng": ""},
        },
        "citizenship_type": random.choice(CITIZENSHIP_TYPES),
        "issue_date":       {"bs": issue, "ad": ""},
        "nationality":      "नेपाली",
    }


def build_gt_old(names, districts, municipalities):
    name = random.choice(names["male"] + names["female"])
    father = random.choice(names["male"])
    dob = random_bs_date(2010, 2055)
    birth = random_address(districts, municipalities)
    cn = random_citizenship_number_old()
    issue = random_bs_issue_date(dob["bs"]["year"]).replace("-", "/")
    return {
        "name":               {"dev": name["dev"],   "eng": ""},
        "gender":             {"dev": random.choice(["पु", "म"]), "eng": ""},
        "citizenship_number": {"dev": cn["dev"],     "eng": ""},
        "dob":                {"bs": dob["bs"],       "ad": {"year": "", "month": "", "day": ""}},
        "birth_address":      birth,
        "current_address":    {"district": {"dev": "", "eng": ""}, "municipality": {"dev": "", "eng": ""}, "ward": ""},
        "parents":            {"father": {"dev": father["dev"], "eng": ""}, "mother": {"dev": "", "eng": ""}},
        "citizenship_type":   random.choice(["वंशज", "जन्म"]),
        "issue_date":         {"bs": issue, "ad": ""},
        "nationality":        "नेपाली",
    }

# ─── FIELD → TEXT MAPPING ─────────────────────────────────────────────────────


def get_field_text(field_name, gt):
    """Returns (display_text, script) for a given field name and ground truth."""
    g = gt
    mapping = {
        "name_dev":               (g.get("name", {}).get("dev", ""),                       "dev"),
        "name_eng":               (g.get("name", {}).get("eng", ""),                       "eng"),
        "gender_dev":             (g.get("gender", {}).get("dev", ""),                     "dev"),
        "gender_eng":             (g.get("gender", {}).get("eng", ""),                     "eng"),
        "nationality":            (g.get("nationality", ""),                               "dev"),
        "citizenship_number_dev": (g.get("citizenship_number", {}).get("dev", ""),         "dev"),
        "citizenship_number_eng": (g.get("citizenship_number", {}).get("eng", ""),         "eng"),
        "citizenship_number":     (g.get("citizenship_number", {}).get("dev", ""),         "dev"),
        "citizenship_type":       (g.get("citizenship_type", ""),                          "dev"),
        "dob_year_bs":            (g.get("dob", {}).get("bs", {}).get("year", ""),          "dev"),
        "dob_month_bs":           (g.get("dob", {}).get("bs", {}).get("month", ""),         "dev"),
        "dob_day_bs":             (g.get("dob", {}).get("bs", {}).get("day", ""),           "dev"),
        "dob_year_ad":            (g.get("dob", {}).get("ad", {}).get("year", ""),          "eng"),
        "dob_month_ad":           (g.get("dob", {}).get("ad", {}).get("month", ""),         "eng"),
        "dob_day_ad":             (g.get("dob", {}).get("ad", {}).get("day", ""),           "eng"),
        "birth_district_dev":     (g.get("birth_address", {}).get("district", {}).get("dev", ""),      "dev"),
        "birth_district_eng":     (g.get("birth_address", {}).get("district", {}).get("eng", ""),      "eng"),
        "birth_municipality_dev": (g.get("birth_address", {}).get("municipality", {}).get("dev", ""),  "dev"),
        "birth_municipality_eng": (g.get("birth_address", {}).get("municipality", {}).get("eng", ""),  "eng"),
        "birth_ward_number_dev":  (g.get("birth_address", {}).get("ward", ""),             "dev"),
        "birth_ward_number_eng":  (str(random.randint(1, 20)),                            "eng"),
        "current_district_dev":   (g.get("current_address", {}).get("district", {}).get("dev", ""),    "dev"),
        "current_district_eng":   (g.get("current_address", {}).get("district", {}).get("eng", ""),    "eng"),
        "current_municipality_dev": (g.get("current_address", {}).get("municipality", {}).get("dev", ""), "dev"),
        "current_municipality_eng": (g.get("current_address", {}).get("municipality", {}).get("eng", ""), "eng"),
        "current_ward_number_dev": (g.get("current_address", {}).get("ward", ""),          "dev"),
        "current_ward_number_eng": (str(random.randint(1, 20)),                           "eng"),
        "father_name_dev":        (g.get("parents", {}).get("father", {}).get("dev", ""),   "dev"),
        "father_name":            (g.get("parents", {}).get("father", {}).get("dev", ""),   "dev"),
        "mother_name_dev":        (g.get("parents", {}).get("mother", {}).get("dev", ""),   "dev"),
        "issue_date_bs":          (g.get("issue_date", {}).get("bs", ""),                  "dev"),
        "issue_date":             (g.get("issue_date", {}).get("bs", ""),                  "dev"),
        "district":               (g.get("birth_address", {}).get("district", {}).get("dev", ""),      "dev"),
        "municipality":           (g.get("birth_address", {}).get("municipality", {}).get("dev", ""),  "dev"),
        "ward_number":            (g.get("birth_address", {}).get("ward", ""),             "dev"),
        "gender":                 (g.get("gender", {}).get("dev", ""),                     "dev"),
        "officer_name":           ("",                                                    "dev"),
        "birth_country_dev":      ("नेपाल",                                              "dev"),
        "birth_country_eng":      ("Nepal",                                               "eng"),
        "husbane/wife_name_dev":  ("",                                                    "dev"),
    }
    return mapping.get(field_name, ("", "dev"))

# ─── TEXT STAMPING (FIXED ALIGNMENT) ─────────────────────────────────────────


def stamp_text_ttf(pil_img, text, bbox, img_w, img_h):
    """
    Stamp text using TTF font with FIXED alignment.

    Key fix: using anchor="lm" (left-middle) so text is centered vertically
    in the bbox regardless of Devanagari glyph ascenders/descenders.
    Font size is computed PER FIELD based on bbox height.
    """
    if not text or not text.strip():
        return pil_img

    x1 = int((bbox["x"] / 100.0) * img_w)
    y1 = int((bbox["y"] / 100.0) * img_h)
    x2 = int(((bbox["x"] + bbox["width"]) / 100.0) * img_w)
    y2 = int(((bbox["y"] + bbox["height"]) / 100.0) * img_h)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 2 or box_h <= 2:
        return pil_img

    # Per-field font size based on bbox height
    font_size = max(10, int(box_h * FONT_HEIGHT_FILL))
    is_dev = any(0x0900 <= ord(c) <= 0x097F for c in text)
    font_key = "dev" if is_dev else "eng"
    font_path = os.path.join(FONTS_DIR, font_key,
                             "NotoSansDevanagari-Regular.ttf" if is_dev else "NotoSans-Regular.ttf")
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(pil_img)

    # Anchor point: left edge, vertical middle of bbox
    # anchor="lm" means: the left edge of the text baseline is at (x, y),
    # and "m" centers it vertically around y.
    text_x = x1 + 2
    text_y = y1 + box_h // 2   # vertical center of bbox

    # Ink color: slightly randomized dark blue-black (not pure black)
    r = random.randint(0, 40)
    g = random.randint(0, 40)
    b = random.randint(10, 70)

    try:
        draw.text((text_x, text_y), text, font=font,
                  fill=(r, g, b), anchor="lm")
    except TypeError:
        # Older Pillow doesn't support anchor — fall back to manual centering
        try:
            bbox_text = draw.textbbox((0, 0), text, font=font)
            th = bbox_text[3] - bbox_text[1]
            draw.text((text_x, y1 + (box_h - th) // 2),
                      text, font=font, fill=(r, g, b))
        except Exception:
            pass

    return pil_img


def stamp_text_handwritten(pil_img, text, bbox, img_w, img_h, hw_index):
    """
    Overlay a handwritten word image onto the document at bbox position.
    Composites RGBA word image onto the document background.
    Falls back to TTF if word not in cache.
    """
    # Try to find word in cache (try full text, then word by word)
    word_img_path = None
    candidates = hw_index.get(text)
    if candidates:
        word_img_path = random.choice(candidates)

    if word_img_path is None or not os.path.exists(word_img_path):
        # Fall back to TTF
        return stamp_text_ttf(pil_img, text, bbox, img_w, img_h)

    # Load handwritten word RGBA image
    hw_img = cv2.imread(word_img_path, cv2.IMREAD_UNCHANGED)
    if hw_img is None or hw_img.shape[2] < 4:
        return stamp_text_ttf(pil_img, text, bbox, img_w, img_h)

    # Target region on document
    x1 = int((bbox["x"] / 100.0) * img_w)
    y1 = int((bbox["y"] / 100.0) * img_h)
    x2 = int(((bbox["x"] + bbox["width"]) / 100.0) * img_w)
    y2 = int(((bbox["y"] + bbox["height"]) / 100.0) * img_h)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 2 or box_h <= 2:
        return pil_img

    # Scale word image to fit bbox height
    hw_h, hw_w = hw_img.shape[:2]
    scale = box_h / hw_h
    new_w = min(box_w, int(hw_w * scale))
    new_h = box_h
    hw_resized = cv2.resize(hw_img, (new_w, new_h),
                            interpolation=cv2.INTER_AREA)

    # Convert PIL to numpy for compositing
    doc_np = np.array(pil_img)

    # Composite: alpha blend handwritten over document
    hw_bgr = hw_resized[:, :, :3]
    hw_a = hw_resized[:, :, 3:4].astype(np.float32) / 255.0

    # Region on document (clamp to bounds)
    rx1 = x1
    ry1 = y1 + (box_h - new_h) // 2
    rx2 = min(img_w, rx1 + new_w)
    ry2 = min(img_h, ry1 + new_h)
    crop_w = rx2 - rx1
    crop_h = ry2 - ry1
    if crop_w <= 0 or crop_h <= 0:
        return pil_img

    doc_region = doc_np[ry1:ry2, rx1:rx2].astype(np.float32)
    hw_crop_bgr = hw_bgr[:crop_h, :crop_w].astype(np.float32)
    hw_crop_a = hw_a[:crop_h, :crop_w]

    blended = (hw_crop_a * hw_crop_bgr + (1 - hw_crop_a)
               * doc_region).astype(np.uint8)
    doc_np[ry1:ry2, rx1:rx2] = blended

    return Image.fromarray(doc_np)

# ─── AUGMENTATION ─────────────────────────────────────────────────────────────


def augment(cv_img):
    if random.random() < AUG_ROTATION_PROB:
        angle = random.uniform(-AUG_ROTATION_MAX, AUG_ROTATION_MAX)
        h, w = cv_img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if random.random() < AUG_BRIGHTNESS_PROB:
        cv_img = cv2.convertScaleAbs(cv_img,
                                     alpha=random.uniform(0.75, 1.25),
                                     beta=random.randint(-20, 20))
    if random.random() < AUG_BLUR_PROB:
        k = random.choice([3, 5])
        cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)
    if random.random() < AUG_NOISE_PROB:
        noise = np.random.normal(0, random.uniform(
            2, 8), cv_img.shape).astype(np.int16)
        cv_img = np.clip(cv_img.astype(np.int16) +
                         noise, 0, 255).astype(np.uint8)
    return cv_img

# ─── MAIN GENERATION LOOP ─────────────────────────────────────────────────────


def generate_for_source(source, names, districts, municipalities, hw_index):
    name = source["name"]
    templates_dir = source["templates_dir"]
    layout_path = source["layout_map"]
    output_dir = source["output_dir"]
    count = source["count"]
    doc_type = source["doc_type"]
    hw_fields = set(source.get("hw_fields", []))
    hw_prob = source.get("hw_probability", 0.0)

    print(f"\n{'─'*60}")
    print(f"  Generating: {name}  ({count} samples)")
    print(f"{'─'*60}")

    if not os.path.exists(templates_dir) or not os.listdir(templates_dir):
        print(f"  ERROR: No templates found in {templates_dir}")
        return

    if not os.path.exists(layout_path):
        print(f"  ERROR: Layout map not found: {layout_path}")
        return

    layout_map = load_json(layout_path)
    templates = [
        os.path.join(templates_dir, f)
        for f in os.listdir(templates_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"  Templates:   {len(templates)}")
    print(f"  Fields:      {len(layout_map)}")
    print(f"  HW fields:   {hw_fields if hw_fields else 'none'}")
    print(f"  HW prob:     {hw_prob*100:.0f}%")

    images_dir = os.path.join(output_dir, "images")
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.jsonl")

    generated = errors = 0
    hw_used = 0

    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for i in tqdm(range(1, count + 1), desc=f"  {name}"):
            try:
                template_path = random.choice(templates)
                cv_template = cv2.imread(template_path)
                if cv_template is None:
                    errors += 1
                    continue

                img_h, img_w = cv_template.shape[:2]

                gt = build_gt_new(names, districts, municipalities) \
                    if doc_type == "new" \
                    else build_gt_old(names, districts, municipalities)

                pil_img = Image.fromarray(
                    cv2.cvtColor(cv_template, cv2.COLOR_BGR2RGB))
                use_hw_this = hw_index and random.random() < hw_prob
                if use_hw_this:
                    hw_used += 1

                # Stamp each field
                for field_name, bbox in layout_map.items():
                    text, _ = get_field_text(field_name, gt)
                    if not text:
                        continue

                    if use_hw_this and field_name in hw_fields and hw_index:
                        # Try handwritten overlay for this field
                        # Split multi-word text — look up each word separately
                        words = text.split()
                        if len(words) == 1:
                            pil_img = stamp_text_handwritten(
                                pil_img, text, bbox, img_w, img_h, hw_index)
                        else:
                            # For multi-word, try to find the whole phrase first
                            if text in hw_index:
                                pil_img = stamp_text_handwritten(
                                    pil_img, text, bbox, img_w, img_h, hw_index)
                            else:
                                # Fall back to TTF for multi-word not in cache
                                pil_img = stamp_text_ttf(
                                    pil_img, text, bbox, img_w, img_h)
                    else:
                        pil_img = stamp_text_ttf(
                            pil_img, text, bbox, img_w, img_h)

                # Augment
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                cv_img = augment(cv_img)

                # Save
                out_filename = f"synth_{i:05d}.jpg"
                out_path = os.path.join(images_dir, out_filename)
                cv2.imwrite(out_path, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

                meta_f.write(json.dumps({
                    "image":        f"images/{out_filename}",
                    "ground_truth": json.dumps(gt, ensure_ascii=False),
                }, ensure_ascii=False) + "\n")
                generated += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n  Error on sample {i}: {e}")

    print(f"\n  Done:  {generated} generated, {errors} errors")
    print(
        f"  HW overlays used: {hw_used} ({hw_used/max(1, generated)*100:.1f}%)")
    print(f"  Images  → {images_dir}/")
    print(f"  Labels  → {metadata_path}")


def main():
    print("\nSynthetic Document Generator v2")
    print("(Fixed alignment + handwriting overlay)\n")

    for f in ["nepali_names.json", "districts.json", "municipalities.json"]:
        if not os.path.exists(os.path.join(ASSETS_DIR, f)):
            print(f"ERROR: Missing {f} in {ASSETS_DIR}/")
            return

    names = load_json(os.path.join(ASSETS_DIR, "nepali_names.json"))
    districts = load_json(os.path.join(ASSETS_DIR, "districts.json"))
    municipalities = load_json(os.path.join(ASSETS_DIR, "municipalities.json"))
    hw_index = load_hw_index()

    print(f"Names: {len(names['male'])}M + {len(names['female'])}F  |  "
          f"Districts: {len(districts)}  |  "
          f"Municipality pools: {len(municipalities)}  |  "
          f"HW words: {len(hw_index)}")

    for source in SOURCES:
        generate_for_source(source, names, districts, municipalities, hw_index)

    print("\n\nAll done!")
    print("Next: python merge_dataset.py  →  then  python train.py")


if __name__ == "__main__":
    main()
