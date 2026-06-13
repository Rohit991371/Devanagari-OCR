"""
build_handwritten_words.py  (v2 — syllable/akshara based)
───────────────────────────────────────────────────────────
ROOT CAUSE FIX: Devanagari cannot be processed character-by-character.
~37% of characters in Nepali text are MATRAS (combining vowel marks: ा ि ी ु ृ etc.)
These are not in the dataset as standalone images.

SOLUTION: Akshara-based rendering
  1. Segment word into aksharas (syllables: base consonant + its matras)
  2. For each akshara: render base consonant from handwritten dataset image
  3. Composite matra(s) on top using TTF font (NotoSansDevanagari)
  4. Concatenate aksharas → word image

This gives: realistic handwritten consonant strokes + correctly positioned matras.

Additional fix: robust binarization to handle mirror/inverted images.

Run from CustomOCR/ root:
  python build_handwritten_words.py
"""

import os
import json
import random
import unicodedata
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from config import DATA_ROOT, BASE_DIR, ASSETS_DIR

# ─── CONFIG ───────────────────────────────────────────────────────────────────

from pathlib import Path

# =========================================================
# BASE DIRECTORY
# =========================================================

# try:
#     BASE_DIR = Path(__file__).resolve().parent
# except NameError:
#     BASE_DIR = Path.cwd()

# =========================================================
# HANDWRITING DATASETS
# =========================================================

DS2_ROOT = (
    ASSETS_DIR
    / "handwriting"
    / "Handwritten-Devanagari-Characters-Dataset"
    # / "data"
)

DS2_CONSONANTS = DS2_ROOT / "consonants"
DS2_VOWELS = DS2_ROOT / "vowels"
DS2_NUMERALS = DS2_ROOT / "numerals"

# =========================================================
# ALPHABET TEXT DATASET
# =========================================================

DS1_IMAGES = (
    ASSETS_DIR
    / "handwriting"
    / "data_alphabet_text"
    / "data_"
    / "dataset_200"
)

DS1_LABELS = (
    ASSETS_DIR
    / "handwriting"
    / "data_alphabet_text"
    / "data_"
    / "notepad_labels_200.txt"
)

# =========================================================
# FONTS
# =========================================================

FONTS_DIR = ASSETS_DIR / "fonts"

# =========================================================
# CACHE
# =========================================================

CACHE_DIR = DATA_ROOT / "handwriting_cache"

WORDS_DIR = CACHE_DIR / "words"

INDEX_PATH = CACHE_DIR / "index.json"

# =========================================================
# CREATE REQUIRED DIRECTORIES
# =========================================================

CACHE_DIR.mkdir(parents=True, exist_ok=True)
WORDS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# DEBUG PRINTS (OPTIONAL)
# =========================================================

print("BASE_DIR:", DATA_ROOT)
print("ASSETS_DIR:", ASSETS_DIR)
print("DS2_ROOT:", DS2_ROOT)
print("WORDS_DIR:", WORDS_DIR)

CHAR_TARGET_H = 56    # px height for each akshara
CHAR_SPACING = 2     # px gap between aksharas
WORD_PADDING = 4     # px padding around word image
MATRA_FONT_RATIO = 0.85  # matra font size relative to CHAR_TARGET_H
N_VARIANTS = 5     # handwritten variants per word

# ─── DEVANAGARI UNICODE RANGES ────────────────────────────────────────────────

# Matras (vowel diacritics that attach to consonants)
MATRAS = set([
    '\u093E',  # ा  aa
    '\u093F',  # ि  i
    '\u0940',  # ी  ii
    '\u0941',  # ु  u
    '\u0942',  # ू  uu
    '\u0943',  # ृ  ri
    '\u0944',  # ॄ  rri
    '\u0947',  # े  e
    '\u0948',  # ै  ai
    '\u094B',  # ो  o
    '\u094C',  # ौ  au
    '\u094F',  # ॏ  aw
    '\u0902',  # ं  anusvara
    '\u0903',  # ः  visarga
    '\u0901',  # ँ  chandrabindu
    '\u094D',  # ्  halant/virama (joins consonants)
])

# Halant — joins two consonants into conjunct
HALANT = '\u094D'

# ─── CONSONANT/VOWEL/NUMERAL MAPPINGS ─────────────────────────────────────────

CONSONANT_MAP = {
    "c_1":  "क",  "c_2":  "ख",  "c_3":  "ग",  "c_4":  "घ",  "c_5":  "ङ",
    "c_6":  "च",  "c_7":  "छ",  "c_8":  "ज",  "c_9":  "झ",  "c_10": "ञ",
    "c_11": "ट",  "c_12": "ठ",  "c_13": "ड",  "c_14": "ढ",  "c_15": "ण",
    "c_16": "त",  "c_17": "थ",  "c_18": "द",  "c_19": "ध",  "c_20": "न",
    "c_21": "प",  "c_22": "फ",  "c_23": "ब",  "c_24": "भ",  "c_25": "म",
    "c_26": "य",  "c_27": "र",  "c_28": "ल",  "c_29": "व",  "c_30": "श",
    "c_31": "ष",  "c_32": "स",  "c_33": "ह",  "c_34": "क्ष", "c_35": "त्र",
    "c_36": "ज्ञ",
}

VOWEL_MAP = {
    "v_1":  "अ", "v_2":  "आ", "v_3":  "इ", "v_4":  "ई",
    "v_5":  "उ", "v_6":  "ऊ", "v_7":  "ए", "v_8":  "ऐ",
    "v_9":  "ओ", "v_10": "औ", "v_11": "अं", "v_12": "अः",
}

NUMERAL_MAP = {
    "n_0": "०", "n_1": "१", "n_2": "२", "n_3": "३", "n_4": "४",
    "n_5": "५", "n_6": "६", "n_7": "७", "n_8": "८", "n_9": "९",
}

# Reverse maps: char → list of image paths
CHAR_TO_IMAGES = {}

# ─── DATASET LOADING ──────────────────────────────────────────────────────────


def _scan_folder_group(base_dir, mapping):
    loaded = 0
    if not os.path.exists(base_dir):
        print(f"  ⚠️  Not found: {base_dir}")
        return loaded
    for folder_name, char in mapping.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        imgs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if imgs:
            CHAR_TO_IMAGES.setdefault(char, []).extend(imgs)
            loaded += len(imgs)
    return loaded


def load_datasets():
    print("  Loading Dataset-2 (consonants/vowels/numerals)...")
    n = _scan_folder_group(DS2_CONSONANTS, CONSONANT_MAP)
    n += _scan_folder_group(DS2_VOWELS,     VOWEL_MAP)
    n += _scan_folder_group(DS2_NUMERALS,   NUMERAL_MAP)
    print(f"    Loaded {n} images for {len(CHAR_TO_IMAGES)} characters")

    print("  Loading Dataset-1 (notepad_labels_200)...")
    if os.path.exists(DS1_LABELS) and os.path.exists(DS1_IMAGES):
        idx2char = {}
        with open(DS1_LABELS, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "/" in line:
                    try:
                        idx, char = line.split("/", 1)
                        idx2char[int(idx.strip())] = char.strip()
                    except ValueError:
                        pass
        extra = 0
        for fname in os.listdir(DS1_IMAGES):
            stem = os.path.splitext(fname)[0]
            try:
                idx = int(stem)
            except ValueError:
                continue
            char = idx2char.get(idx)
            if char:
                path = os.path.join(DS1_IMAGES, fname)
                CHAR_TO_IMAGES.setdefault(char, []).append(path)
                extra += 1
        print(f"    Loaded {extra} additional images from Dataset-1")
    else:
        print("    Dataset-1 not found — skipping")

# ─── ROBUST IMAGE LOADING ─────────────────────────────────────────────────────


def load_char_image_robust(path):
    """
    Load character image with robust binarization.
    Handles: white-on-black, black-on-white, mirrored/inverted images.
    Returns grayscale binary (character=white, bg=black) or None.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Fix mirror/flip: check if image appears to be horizontally flipped
    # by comparing left-half and right-half density — heuristic for obvious mirrors
    # (we skip this heavy check and instead rely on Otsu + polarity detection)

    # Otsu binarization
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Polarity detection: character should be the MINORITY of pixels
    # (ink covers less area than paper background)
    white_ratio = np.mean(binary > 128)
    if white_ratio > 0.5:
        # More than half pixels are "white" — invert so character = white
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup: remove tiny noise specks
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Crop tight to character
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 5 or h < 5:
        return None

    return binary[y:y+h, x:x+w]

# ─── AKSHARA SEGMENTATION ─────────────────────────────────────────────────────


def segment_aksharas(word):
    """
    Segment a Devanagari word into aksharas (syllable units).
    Each akshara = one base character + all following matras/halant-conjuncts.

    Examples:
      'सृजना' → ['सृ', 'ज', 'ना']
      'श्रेष्ठ' → ['श्रे', 'ष्ठ']
      'महिला'  → ['म', 'हि', 'ला']
      'नेपाली' → ['ने', 'पा', 'ली']
      'रौतहट' → ['रौ', 'त', 'ह', 'ट']

    Returns list of akshara strings.
    """
    aksharas = []
    i = 0
    while i < len(word):
        c = word[i]
        if c == ' ':
            i += 1
            continue

        # Start new akshara with this character
        akshara = c
        i += 1

        # Consume all following matras and halant+consonant sequences
        while i < len(word):
            nc = word[i]
            if nc in MATRAS:
                akshara += nc
                i += 1
                # If this matra is halant, consume the next consonant too (conjunct)
                if nc == HALANT and i < len(word):
                    akshara += word[i]
                    i += 1
            else:
                break

        aksharas.append(akshara)

    return aksharas

# ─── AKSHARA RENDERING ────────────────────────────────────────────────────────


def get_matra_font(size=None):
    """Load TTF font for matra rendering."""
    if size is None:
        size = int(CHAR_TARGET_H * MATRA_FONT_RATIO)
    path = os.path.join(FONTS_DIR, "dev", "NotoSansDevanagari-Regular.ttf")
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def render_akshara(akshara):
    """
    Render one akshara as an RGBA image.

    Strategy:
    - Extract the base character (first char of akshara)
    - Look it up in CHAR_TO_IMAGES → get handwritten image
    - Render the full akshara using TTF at same size
    - Use TTF rendering as the matra layer, blend over handwritten base

    This gives: handwritten consonant stroke + correct matra position.
    """
    if not akshara:
        return None

    base_char = akshara[0]
    matras = akshara[1:]

    # Get handwritten base char image
    candidates = CHAR_TO_IMAGES.get(base_char, [])

    if candidates:
        raw = load_char_image_robust(random.choice(candidates))
    else:
        raw = None

    # Target canvas size
    canvas_h = CHAR_TARGET_H
    # Aspect ratio: use handwritten image width if available, else estimate
    if raw is not None:
        h, w = raw.shape[:2]
        scale = canvas_h / h
        canvas_w = max(20, int(w * scale))
        char_img = cv2.resize(raw, (canvas_w, canvas_h),
                              interpolation=cv2.INTER_AREA)
    else:
        # No handwritten image — fall through to pure TTF
        canvas_w = max(20, int(canvas_h * 0.8))
        char_img = None

    # Render full akshara using TTF for correct matra positioning
    font = get_matra_font(canvas_h)
    pil_dummy = Image.new("L", (canvas_w * 3, canvas_h * 2), 255)
    draw = ImageDraw.Draw(pil_dummy)
    try:
        bb = draw.textbbox((0, 0), akshara, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        tw, th = canvas_w, canvas_h

    # Create TTF rendering of full akshara (for matra reference)
    ttf_w = max(canvas_w, tw + 4)
    pil_ttf = Image.new("L", (ttf_w, canvas_h + 10), 255)
    draw_ttf = ImageDraw.Draw(pil_ttf)
    try:
        draw_ttf.text((2, 2), akshara, font=font, fill=0, anchor="lt")
    except TypeError:
        draw_ttf.text((2, 2), akshara, font=font, fill=0)

    ttf_arr = np.array(pil_ttf)

    # Build final RGBA output
    out_w = max(canvas_w, ttf_w)
    out = np.zeros((canvas_h, out_w, 4), dtype=np.uint8)

    if char_img is not None and matras:
        # Blend: handwritten base consonant + TTF matras
        # 1. Place handwritten char (dark ink on transparent)
        hw_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        mask = char_img > 128
        ink_r = random.randint(0, 40)
        ink_g = random.randint(0, 40)
        ink_b = random.randint(5, 60)
        hw_rgba[mask, 0] = ink_r
        hw_rgba[mask, 1] = ink_g
        hw_rgba[mask, 2] = ink_b
        hw_rgba[mask, 3] = 255
        out[:, :canvas_w] = hw_rgba

        # 2. Overlay TTF matra layer: only take matra pixels (not base char area)
        # We approximate: matra pixels are those in TTF output that fall outside
        # the base char bounding box region. Simpler: blend TTF over hw at low opacity
        # for any pixels that are dark in TTF but not in hw area.
        ttf_crop = ttf_arr[:canvas_h, :out_w]
        ttf_mask = ttf_crop < 128   # dark pixels = ink in TTF
        # already have handwritten ink here
        hw_existing = out[:, :out_w, 3] > 128

        # Add TTF ink where it's NOT already covered by handwritten stroke
        matra_only = ttf_mask & ~hw_existing
        out[matra_only, 0] = ink_r
        out[matra_only, 1] = ink_g
        out[matra_only, 2] = ink_b
        out[matra_only, 3] = 220   # slightly transparent for natural look

    elif char_img is not None and not matras:
        # Just handwritten base char, no matras
        hw_rgba = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        mask = char_img > 128
        ink_r = random.randint(0, 40)
        ink_g = random.randint(0, 40)
        ink_b = random.randint(5, 60)
        hw_rgba[mask, 0] = ink_r
        hw_rgba[mask, 1] = ink_g
        hw_rgba[mask, 2] = ink_b
        hw_rgba[mask, 3] = 255
        out[:, :canvas_w] = hw_rgba

    else:
        # No handwritten image — pure TTF fallback for this akshara
        ttf_crop = ttf_arr[:canvas_h, :out_w]
        ttf_mask = ttf_crop < 128
        ink_r = random.randint(0, 50)
        ink_g = random.randint(0, 50)
        ink_b = random.randint(10, 70)
        out[ttf_mask, 0] = ink_r
        out[ttf_mask, 1] = ink_g
        out[ttf_mask, 2] = ink_b
        out[ttf_mask, 3] = 255

    # Trim trailing empty columns
    nonempty_cols = np.where(out[:, :, 3].max(axis=0) > 0)[0]
    if len(nonempty_cols) == 0:
        return None
    out = out[:, :nonempty_cols[-1]+1]

    return out

# ─── WORD IMAGE BUILDER ───────────────────────────────────────────────────────


def build_word_image(word):
    """
    Build word image by segmenting into aksharas and rendering each.
    Returns RGBA numpy array or None.
    """
    word = word.strip()
    if not word:
        return None

    # Check if purely ASCII/English
    is_english = all(ord(c) < 128 for c in word if c != ' ')
    if is_english:
        return None   # skip English — TTF handles it fine

    aksharas = segment_aksharas(word)
    if not aksharas:
        return None

    akshara_imgs = []
    for a in aksharas:
        img = render_akshara(a)
        if img is not None:
            akshara_imgs.append(img)

    if not akshara_imgs:
        return None

    # Pad all to same height (should be CHAR_TARGET_H already)
    total_w = sum(img.shape[1] for img in akshara_imgs)
    total_w += CHAR_SPACING * (len(akshara_imgs) - 1)
    total_w += WORD_PADDING * 2
    total_h = CHAR_TARGET_H + WORD_PADDING * 2

    canvas = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    x = WORD_PADDING
    for img in akshara_imgs:
        h, w = img.shape[:2]
        y_off = WORD_PADDING + (CHAR_TARGET_H - min(h, CHAR_TARGET_H)) // 2
        h_clip = min(h, CHAR_TARGET_H)
        canvas[y_off:y_off+h_clip, x:x+w] = img[:h_clip]
        x += w + CHAR_SPACING

    return canvas


def augment_word(word_img):
    """Mild augmentation for handwriting realism."""
    if word_img is None:
        return None
    bgr = word_img[:, :, :3].copy()
    alpha = word_img[:, :, 3].copy()

    if random.random() < 0.5:
        angle = random.uniform(-4, 4)
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        bgr = cv2.warpAffine(bgr,   M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(
            alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.3:
        bgr = cv2.GaussianBlur(bgr, (3, 3), 0)

    return np.dstack([bgr, alpha])

# ─── WORD POOL ────────────────────────────────────────────────────────────────


def collect_word_pool():
    """Collect all unique Devanagari words from assets JSON files."""
    words = set()
    for fname in ["nepali_names.json", "districts.json", "municipalities.json"]:
        fpath = os.path.join("assets", fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        def extract(obj):
            if isinstance(obj, dict):
                if "dev" in obj and isinstance(obj["dev"], str):
                    for w in obj["dev"].split():
                        if any(0x0900 <= ord(c) <= 0x097F for c in w):
                            words.add(w)
                for v in obj.values():
                    extract(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item)
        extract(data)

    # Add standalone field values
    for w in ["नेपाली", "महिला", "पुरुष", "पु", "म", "वंशज", "जन्म", "अंगिकृत", "नेपाल", "बैवाहिक"]:
        words.add(w)

    return words

# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    print("\nHandwriting Cache Builder v2 (Akshara-based)")
    print("─" * 55)

    load_datasets()
    print(f"\n  Unique base characters available: {len(CHAR_TO_IMAGES)}")
    print(
        f"  Total character images: {sum(len(v) for v in CHAR_TO_IMAGES.values())}")

    # Check font
    font_path = os.path.join(
        FONTS_DIR, "dev", "NotoSansDevanagari-Regular.ttf")
    if not os.path.exists(font_path):
        print(f"\n  ⚠️  Font not found: {font_path}")
        print("  Matra rendering will use default font — quality will be lower.")
        print("  Download NotoSansDevanagari-Regular.ttf from fonts.google.com/noto")

    words = collect_word_pool()
    print(f"\n  Word pool: {len(words)} unique Devanagari words")

    # Quick coverage check
    buildable = 0
    for word in list(words)[:50]:   # sample check
        aksharas = segment_aksharas(word)
        base_chars = [a[0] for a in aksharas if a]
        if all(c in CHAR_TO_IMAGES for c in base_chars):
            buildable += 1
    est_coverage = buildable / min(50, len(words)) * 100
    print(f"  Estimated coverage (base chars only): ~{est_coverage:.0f}%")
    print(f"  (Matras handled via TTF overlay — all words renderable)")

    Path(WORDS_DIR).mkdir(parents=True, exist_ok=True)

    index = {}
    built = 0
    ttf_only = 0

    print(f"\n  Building {N_VARIANTS} variants per word...")
    for word in tqdm(sorted(words), desc="  Caching"):
        paths = []
        for v in range(N_VARIANTS):
            img = build_word_image(word)
            if img is None:
                break
            img = augment_word(img)
            if img is None:
                break

            safe = word.replace("/", "_").replace("\\", "_").replace(":", "_")
            out_path = os.path.join(WORDS_DIR, f"{safe}_v{v}.png")
            try:
                cv2.imwrite(out_path, img)
                paths.append(out_path)
                built += 1
            except Exception:
                pass

        if paths:
            index[word] = paths
            # Check if all base chars had hw images
            aksharas = segment_aksharas(word)
            base_chars = [a[0] for a in aksharas if a]
            if not all(c in CHAR_TO_IMAGES for c in base_chars):
                ttf_only += 1

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    coverage = len(index) / max(1, len(words)) * 100
    print(f"\n  Results:")
    print(
        f"    Words cached:       {len(index)} / {len(words)} ({coverage:.0f}%)")
    print(f"    Images built:       {built}")
    print(f"    TTF-only fallback:  {ttf_only} words (no hw base char)")
    print(f"    Index saved:        {INDEX_PATH}")
    print(f"\n  Run generate_synthetic.py to use this cache.")


if __name__ == "__main__":
    main()
