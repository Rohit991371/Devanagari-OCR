# Devanagari OCR — Nepali Citizenship Document Extraction

Production-grade, OCR-free document intelligence pipeline for Nepali citizenship
certificates (new and old format). Given a photo or scan of a document, the system
outputs a structured JSON record — name, DOB (BS/AD), citizenship number, addresses,
parents, issue date — using a fine-tuned Donut (`naver-clova-ix/donut-base`) model.

Code and data are kept separate on purpose: this repository holds all scripts,
assets, and fonts; the actual document images, annotations, and generated
synthetic data live in a separate `Data/` folder (see [Data layout](#data-layout))
that is synced via Google Drive and pointed to with an environment variable.

---

## Why Donut

Most OCR pipelines are two-stage: detect text regions, then recognize characters
in each region, then run a separate parser to structure the result. Donut
collapses this into one model: a Swin Transformer encoder reads the raw image,
a BART-style autoregressive decoder cross-attends to the encoder's patch
embeddings and generates the *entire structured output* as a single token
sequence — no bounding-box detection step, no separate recognition step.

The output schema is expressed as XML-style tags, one pair per field, in a
fixed order:

```
<s_nepali_citizenship>
  <s_name_dev>सृजना रौनीयार</s_name_dev>
  <s_name_eng>SRIJANA RAUNIYAR</s_name_eng>
  <s_gender_dev>महिला</s_gender_dev>
  ...
</s_nepali_citizenship>
```

These tags are added to the tokenizer as special tokens, so the model emits
`<s_name_dev>` as a single token rather than spelling it out character by
character. The decoder learns a fixed schema — every field is always present,
even as an empty tag, which keeps the output structure predictable and easy
to parse downstream.

---

## Pipeline overview

### 1. Data preparation (real documents → training-ready format)

```
Real scans (65: 40 new + 25 old)
        │
        ▼
Label Studio annotation           bounding box + transcribed text per field
        │
        ├──────────────────────────────┐
        ▼                              ▼
convert_to_donut.py             extract_layout_map.py + create_templates.py
(real ground truth JSON)        (field bbox %, inpainted blank templates)
        │                              │
        │                              ▼
        │                     generate_synthetic.py
        │                     (typed + handwritten overlay,
        │                      camera-capture augmentation)
        │                              │
        └──────────────┬───────────────┘
                        ▼
                merge_dataset.py
        (all real → train; synthetic → 80/10/10 split)
                        │
                        ▼
        Data/processed/{train,val,test}.jsonl
```

- **Label Studio** annotations link a rectangle (`rectanglelabels`) to a text
  transcription (`textarea`) via a shared region `id`. `convert_to_donut.py`
  walks this structure and normalizes it into the nested ground-truth schema
  used everywhere else in the pipeline.
- **`extract_layout_map.py`** aggregates bounding boxes across all annotated
  documents into a median position + spread (`x_std`/`y_std`) per field, in
  percentage coordinates (resolution-independent).
- **`create_templates.py`** uses OpenCV TELEA inpainting to remove the
  printed/handwritten values from real documents, leaving a blank template
  that preserves paper texture and static printed labels.
- **`generate_synthetic.py`** stamps freshly generated field values onto
  those blank templates: typed values via Noto Sans Devanagari/Noto Sans
  (with per-field font sizing and vertical centering to handle Devanagari's
  variable ascenders/descenders), or handwritten values by concatenating
  segmented character images from two Kaggle handwriting datasets — used
  for the old-format fields, since those documents are genuinely
  handwritten in real life. Camera-capture noise (blur, brightness, rotation,
  noise) is applied afterward via OpenCV.
- **`build_handwritten_words.py`** is a one-time preprocessing step that
  builds the character-image cache and word index used above.

### 2. Training

```
train/val/test.jsonl
        │
        ▼
Phase 1 — synthetic pretrain     5 epochs, lr 3e-5
        │                        learns Devanagari glyphs + field schema
        ▼
Phase 2 — real fine-tune         2 epochs, lr 5e-6
        │                        adapts to real scan noise/variation
        ▼
   model_final/
```

Both phases use `VisionEncoderDecoderModel` with gradient checkpointing,
gradient accumulation, cosine LR scheduling with warmup, and per-epoch
checkpointing (best-by-val-loss).

### 3. Evaluation and postprocessing

```
model_final/
        │
        ▼
   evaluate.py         CER per field + full-record exact match, on test.jsonl
        │
        ▼
   postprocess.py       regex validation (BS dates, citizenship number format,
                         gender normalization) + per-field confidence scores
```

---

## Known issues (as of this analysis)

If you're getting poor results after training, check these first — in order
of expected impact:

1. **`train.py` `CitizenshipDataset.__getitem__` builds `decoder_input_ids`
   incorrectly.** It tokenizes only the task-start token and pads the rest
   with `PAD`, instead of passing the shifted label sequence for teacher
   forcing. In practice this means the decoder never sees correct
   token-by-token context during training. Fix: drop the manual
   `decoder_input_ids` construction entirely and call
   `model(pixel_values=pixel_values, labels=labels)` — let
   `VisionEncoderDecoderModel` derive `decoder_input_ids` from `labels`
   internally via `decoder_start_token_id`.

2. **`merge_dataset.py`'s `synth_old` source has a path bug.** Its
   `image_base` is `DATA_ROOT / "Synthetic" / "citizenship_old"`, missing
   the `"Storage"` segment that every other source uses. This silently
   drops all old-format synthetic images from training (counted under
   `bad_image` in the load summary — check that number after a re-run).

3. **`evaluate.py`'s `xml_to_flat` only reverses the first underscore**
   (`tag.replace("_", ".", 1)`), so any field nested more than one level
   deep (`dob.bs.year`, `citizenship_number.dev`, `birth_address.district.dev`)
   is looked up under the wrong key and scores as a miss even when the
   model got it right. `postprocess.py` avoids this by using a consistent
   first-split scheme on both sides — `evaluate.py`'s parser should follow
   the same convention, or field metrics should be treated as unreliable
   until fixed.

4. **`api.py` is currently a duplicate of `postprocess.py`**, not a FastAPI
   app — there's no model-loading or `/extract` route yet, despite the
   filename.

5. **Image size** was reduced from Donut's default 1280×960 to 960×720 to
   fit 8GB VRAM. Reasonable tradeoff, but worth revisiting per-field CER on
   small/dense text (citizenship number, ward numbers) once the training
   bug above is fixed — those fields are most sensitive to resolution.

---

## Data layout

Code (this repo) and data live in sibling directories:

```
<parent>/
├── Devanagari-OCR/         ← this repo
└── Data/
    └── Storage/
        ├── Citizenship_new/
        │   ├── Annotations/{new_doc_annotation.json, metadata.jsonl}
        │   └── Docs/
        ├── Citizenship_old/
        │   ├── Annotations/{old_doc_annotation.json, metadata.jsonl}
        │   └── Docs/
        └── Synthetic/
            ├── citizenship_new/{images/, metadata.jsonl}
            └── citizenship_old/{images/, metadata.jsonl}
    └── processed/            ← written by merge_dataset.py
    └── checkpoints/          ← written by train.py
    └── model_final/          ← written by train.py
    └── evaluation/           ← written by evaluate.py
```

`config.py` resolves `DATA_ROOT` from the `DATA_ROOT` environment variable,
defaulting to `<repo_parent>/Data`. Set this to wherever your synced Drive
folder lands, e.g.:

```bash
export DATA_ROOT="/content/drive/MyDrive/CustomOCR/Data"
```

---

## Running the pipeline

Each stage can be run standalone or via the orchestrator:

```bash
pip install -r requirements.txt

python build_handwritten_words.py    # one-time: build handwriting cache
python convert_to_donut.py           # real annotations -> metadata.jsonl
python generate_synthetic.py         # generate synthetic documents
python merge_dataset.py              # build train/val/test.jsonl
python train.py                      # 2-phase Donut fine-tuning
python evaluate.py                   # CER + exact-match report
```

Or all at once:

```bash
python pipeline.py all
# python pipeline.py train      # run a single stage
```

**Requirements**: GPU strongly recommended (8GB VRAM minimum with batch
size 1 + gradient checkpointing; CPU training will take days). Set
`HF_TOKEN` in a `.env` file if pulling gated models.

---

## Tech stack

| Component | Choice | Why |
|---|---|---|
| OCR model | Donut (`naver-clova-ix/donut-base`) | End-to-end, no separate detection/recognition step |
| Annotation | Label Studio | Free, bbox + text, clean JSON export |
| Template generation | OpenCV TELEA inpainting | Preserves paper texture and static printed labels |
| Fonts | Noto Sans Devanagari / Noto Sans | Full Devanagari Unicode coverage, free |
| Handwriting | Concatenated Kaggle character images | Realistic without a GAN; works with limited real data |
| Training | 2-phase curriculum | Synthetic-first pretraining prevents overfitting on 65 real docs |

---

## Roadmap

- [ ] Fix `decoder_input_ids` construction in `train.py`
- [ ] Fix `synth_old` path in `merge_dataset.py`
- [ ] Align `evaluate.py`'s tag parsing with `postprocess.py`'s convention
- [ ] Re-run training and confirm loss/metrics behave sanely
- [ ] Build an actual FastAPI app in `api.py` (`/extract`, `/extract/batch`, `/health`)
- [ ] Extend to Nepali passports, PAN card, voter ID(Future Plan)
