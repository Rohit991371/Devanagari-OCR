"""
train.py
─────────
2-phase fine-tuning of Donut (naver-clova-ix/donut-base) on Nepali citizenship documents.

Phase 1: Train on synthetic data (5 epochs, LR=3e-5)
         Goal: learn Devanagari text patterns + document field layout

Phase 2: Fine-tune on ALL data including real docs (2 epochs, LR=5e-6)
         Goal: adapt to real scan quality, noise, variation

The model learns to map:
  document image → structured JSON with all fields

Input:  image
Output: JSON string with name, DOB, address, citizenship_number etc.

Run from CustomOCR/ root:
  pip install torch transformers datasets sentencepiece accelerate tqdm Pillow
  python train.py

Requirements:
  - processed/train.jsonl, val.jsonl (from merge_dataset.py)
  - GPU strongly recommended (CPU training will take days)
  - ~8GB VRAM minimum (reduce BATCH_SIZE to 1 if OOM)
"""

from huggingface_hub import login
from dotenv import load_dotenv
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch
import math
import re
import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(HF_TOKEN)
    print("  HuggingFace login successful")
else:
    print("  HF_TOKEN not found")

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# PROCESSED_DIR = "processed"
# CHECKPOINT_DIR = "checkpoints"
# FINAL_MODEL_DIR = "model_final"

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
FINAL_MODEL_DIR = BASE_DIR / "model_final"

# Create dirs automatically
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Donut base model
BASE_MODEL = "naver-clova-ix/donut-base"

# Image size Donut expects (height, width)
# Standard Donut: 1280x960. Reduce if OOM.
# IMAGE_SIZE = (1280, 960) # Reduced due to OOM on 8GB VRAM
IMAGE_SIZE = (960, 720)

# Task token — Donut needs a task-specific start token
TASK_START_TOKEN = "<s_nepali_citizenship>"
TASK_END_TOKEN = "</s_nepali_citizenship>"

# ─────────────────────────────────────────────────────────────
# TRAINING PHASE 1
# Synthetic Pretraining
# ─────────────────────────────────────────────────────────────
PHASE1_CONFIG = {
    "name":        "phase1_synthetic",
    "data_filter": ["synth_new", "synth_old"],   # only synthetic
    "epochs":      5,
    "lr":          3e-5,
    "batch_size":  1,  # reduced from 4 to 1 due to OOM on 8GB VRAM
    "grad_accumulation":    4,   # ← simulates batch size of 4
    "warmup_ratio": 0.1,
    "grad_clip":   1.0,
    "save_every":  1,   # save checkpoint every N epochs
}


# ─────────────────────────────────────────────────────────────
# TRAINING PHASE 2
# Real + Synthetic Finetuning
# ─────────────────────────────────────────────────────────────
PHASE2_CONFIG = {
    "name":        "phase2_finetune",
    "data_filter": None,   # all data (real + synthetic)
    "epochs":      2,
    "lr":          5e-6,
    "batch_size":  1,  # reduced from 2 to 1 due to OOM on 8GB VRAM
    "warmup_ratio": 0.2,
    "grad_clip":   1.0,
    "save_every":  1,
}

# Set to True to run a quick sanity check (10 steps per epoch)
DRY_RUN = False

# ─── FIELD ORDERING ───────────────────────────────────────────────────────────
# Consistent field order for XML serialization. Model must learn a fixed schema.

FIELD_ORDER = [
    "name.dev", "name.eng",
    "gender.dev", "gender.eng",
    "nationality",
    "citizenship_number.dev", "citizenship_number.eng",
    "citizenship_type",
    "dob.bs.year", "dob.bs.month", "dob.bs.day",
    "dob.ad.year", "dob.ad.month", "dob.ad.day",
    "birth_address.district.dev", "birth_address.district.eng",
    "birth_address.municipality.dev", "birth_address.municipality.eng",
    "birth_address.ward",
    "current_address.district.dev", "current_address.district.eng",
    "current_address.municipality.dev", "current_address.municipality.eng",
    "current_address.ward",
    "parents.father.dev", "parents.father.eng",
    "parents.mother.dev", "parents.mother.eng",
    "issue_date.bs", "issue_date.ad",
]


# ─── GROUND TRUTH SERIALIZATION ───────────────────────────────────────────────


def flatten_gt(gt_dict):
    """
    Flatten nested ground truth dict into dot-notation keys.
    Example: gt["name"]["dev"] → "name.dev"
    """
    flat = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (str, int, float)):
            flat[prefix] = str(obj)
        elif isinstance(obj, list):
            flat[prefix] = " ".join(str(x) for x in obj)

    _flatten(gt_dict)
    return flat


def gt_to_xml(gt_dict):
    """
    Serialize ground truth dict to XML-tagged string for Donut decoder target.

    Format:
      <s_nepali_citizenship>
        <s_name_dev>सृजना रौनीयार</s_name_dev>
        <s_name_eng>SRIJANA RAUNIYAR</s_name_eng>
        ...
      </s_nepali_citizenship>

    Empty values are still included (model learns to output empty tags).
    Consistent field ordering is critical — model learns a fixed schema.
    """
    flat = flatten_gt(gt_dict)
    parts = [TASK_START_TOKEN]

    for field in FIELD_ORDER:
        value = flat.get(field, "")
        tag = field.replace(".", "_")
        parts.append(f"<s_{tag}>{value}</s_{tag}>")

    parts.append(TASK_END_TOKEN)
    return "".join(parts)


def xml_to_gt(xml_str):
    """
    Parse XML-tagged string back to nested dict.
    Used during evaluation and inference.
    """
    result = {}
    # Extract all tags
    pattern = re.compile(r"<s_([^>]+)>(.*?)</s_\1>", re.DOTALL)
    matches = pattern.findall(xml_str)

    for tag, value in matches:
        if tag in ("nepali_citizenship",):
            continue
        # Convert tag back to dot notation
        key = tag.replace("_", ".", 1) if "_" in tag else tag
        # Walk dotted path and set value
        parts = key.split(".")
        cursor = result
        for i, part in enumerate(parts[:-1]):
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value.strip()

    return result

# ─── DATASET ──────────────────────────────────────────────────────────────────


class CitizenshipDataset(Dataset):
    """
    Loads citizenship document images and prepares Donut inputs.

    For each sample:
      pixel_values     : preprocessed image tensor [3, H, W]
      decoder_input_ids: tokenized prompt (task start token)
      labels           : tokenized target XML string
    """

    def __init__(self, jsonl_path, processor, source_filter=None, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.records = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                # Filter by source if specified
                if source_filter and rec.get("source") not in source_filter:
                    continue

                # Validate image exists
                if not os.path.exists(rec["image_path"]):
                    continue

                # Parse ground truth
                gt_raw = rec["ground_truth"]
                try:
                    gt = json.loads(gt_raw) if isinstance(
                        gt_raw, str) else gt_raw
                except Exception:
                    continue

                self.records.append({
                    "image_path":   rec["image_path"],
                    "ground_truth": gt,
                    "doc_type":     rec.get("doc_type", "unknown"),
                    "source":       rec.get("source", "unknown"),
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load and preprocess image
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            # Return a blank image if load fails
            image = Image.new(
                "RGB", (IMAGE_SIZE[1], IMAGE_SIZE[0]), (255, 255, 255))

        # Donut processor handles resize + normalization
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)   # [3, H, W]

        # Build target XML string
        target_xml = gt_to_xml(rec["ground_truth"])

        # Tokenize target
        tokenized = self.processor.tokenizer(
            target_xml,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)   # [max_length]

        # Replace padding token id with -100 so loss ignores padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Decoder input: task start token (prompt), padded to max_length for collation
        decoder_input_ids = self.processor.tokenizer(
            TASK_START_TOKEN,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values":      pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "labels":            labels,
        }

# ─── MODEL SETUP ──────────────────────────────────────────────────────────────


def build_model_and_processor():
    """
    Load Donut base model and processor.
    Add custom tokens for our task and all Devanagari field tags.
    """
    print("  Loading Donut base model...")
    processor = DonutProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # Set image size
    processor.image_processor.size = {
        "height": IMAGE_SIZE[0],
        "width":  IMAGE_SIZE[1],
    }
    model.config.encoder.image_size = [IMAGE_SIZE[0], IMAGE_SIZE[1]]

    # Build list of new special tokens to add
    new_tokens = [TASK_START_TOKEN, TASK_END_TOKEN]
    for field in FIELD_ORDER:
        tag = field.replace(".", "_")
        new_tokens.append(f"<s_{tag}>")
        new_tokens.append(f"</s_{tag}>")

    # Add tokens that don't exist yet
    existing = set(processor.tokenizer.get_vocab().keys())
    to_add = [t for t in new_tokens if t not in existing]
    if to_add:
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": to_add})
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"  Added {len(to_add)} new tokens to tokenizer")

    # Configure decoder
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        TASK_START_TOKEN
    )

    return model, processor

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────


def train_phase(model, processor, phase_config, device):
    """Run one training phase."""
    name = phase_config["name"]
    epochs = phase_config["epochs"]
    lr = phase_config["lr"]
    batch_size = phase_config["batch_size"]
    warmup_r = phase_config["warmup_ratio"]
    grad_clip = phase_config["grad_clip"]
    save_every = phase_config["save_every"]
    filt = phase_config["data_filter"]

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  epochs={epochs}  lr={lr}  batch={batch_size}")
    print(f"  data_filter={filt if filt else 'all'}")
    print(f"{'─'*60}")

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")

    # 512 tokens × batch_size × decoder layers = significant memory. For your field count (29 fields), the actual XML is typically 200–300 tokens.
    train_ds = CitizenshipDataset(
        train_path, processor, source_filter=filt, max_length=384)

    val_ds = CitizenshipDataset(
        val_path,   processor, source_filter=None, max_length=384)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")

    if len(train_ds) == 0:
        print("  ERROR: No training samples. Check data_filter and processed/train.jsonl")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              # changed pin_memory to conditional
                              num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_r)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── TRAIN ──
        model.train()

        model.gradient_checkpointing_enable()  # Added
        # # This recomputes activations during backward pass instead of storing them.
        # # Costs ~20% more compute time but cuts VRAM by ~40-60%.

        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs} [train]")
        for step, batch in enumerate(pbar):
            if DRY_RUN and step >= 10:
                break

            pixel_values = batch["pixel_values"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = train_loss / max(1, train_steps)

        # ── VALIDATE ──
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc=f"  Epoch {epoch}/{epochs} [val]")):
                if DRY_RUN and step >= 5:
                    break

                pixel_values = batch["pixel_values"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss / max(1, val_steps)

        print(
            f"\n  Epoch {epoch}: train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        # Save checkpoint every N epochs
        if epoch % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:02d}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Track best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best")
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(
                f"  Best model updated (val_loss={best_val_loss:.4f}): {best_path}")

    return os.path.join(checkpoint_dir, "best")

# ─── QUICK INFERENCE TEST ─────────────────────────────────────────────────────


def run_inference_sample(model, processor, device, num_samples=2):
    """
    Run inference on a few val samples to sanity check model output.
    Prints raw XML output and parsed JSON.
    """
    print("\n  Inference sanity check...")
    val_path = os.path.join(PROCESSED_DIR, "val.jsonl")
    if not os.path.exists(val_path):
        return

    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    model.eval()
    for i, rec in enumerate(samples):
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            continue

        pixel_values = processor(
            image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=processor.tokenizer(
                    TASK_START_TOKEN, add_special_tokens=False, return_tensors="pt"
                ).input_ids.to(device),
                max_length=512,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.convert_tokens_to_ids(
                    TASK_END_TOKEN),
            )

        raw_output = processor.tokenizer.decode(
            outputs[0], skip_special_tokens=False)
        parsed = xml_to_gt(raw_output)

        print(f"\n  Sample {i+1}:")
        print(f"  Source: {rec.get('source', 'unknown')}")
        print(f"  Raw output (first 200 chars): {raw_output[:200]}...")
        print(f"  Parsed name: {parsed.get('name', {})}")
        print(
            f"  Parsed citizenship_number: {parsed.get('citizenship_number', {})}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    print("\nDonut Fine-tuning — Nepali Citizenship OCR")
    print("─" * 60)

    # Validate prerequisites
    for split in ["train.jsonl", "val.jsonl"]:
        path = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run merge_dataset.py first.")
            return

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Device: GPU ({torch.cuda.get_device_name(0)})")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (training will be slow — GPU strongly recommended)")

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Load model
    model, processor = build_model_and_processor()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {total_params:.0f}M")

    # ── PHASE 1: Synthetic data ──
    phase1_best = train_phase(model, processor, PHASE1_CONFIG, device)
    print(f"\n  Phase 1 complete. Best checkpoint: {phase1_best}")

    # Load best phase 1 checkpoint for phase 2
    if phase1_best and os.path.exists(phase1_best):
        print(f"\n  Loading best Phase 1 checkpoint for Phase 2...")
        model = VisionEncoderDecoderModel.from_pretrained(
            phase1_best).to(device)

    # ── PHASE 2: All data fine-tune ──
    phase2_best = train_phase(model, processor, PHASE2_CONFIG, device)
    print(f"\n  Phase 2 complete. Best checkpoint: {phase2_best}")

    # Load and save final model
    if phase2_best and os.path.exists(phase2_best):
        model = VisionEncoderDecoderModel.from_pretrained(
            phase2_best).to(device)

    Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)
    print(f"\n  Final model saved: {FINAL_MODEL_DIR}/")

    # Sanity check
    torch.cuda.empty_cache()  # Clear VRAM before inference test
    run_inference_sample(model, processor, device)

    print("\n  Training complete!")
    print("  Next: python evaluate.py")


if __name__ == "__main__":
    main()
