import json
import os
from collections import defaultdict
import re

# INPUT_JSON = "D:\\Web Dev\\Custome OCR\\CustomOCR\\Data\\Citizenship_old\\Annotations\\old_doc_annotation.json"
# IMAGE_DIR = "D:\\Web Dev\\Custome OCR\\CustomOCR\\Data\\Citizenship_old\\Docs"
# OUTPUT_JSONL = "D:\\Web Dev\\Custome OCR\\CustomOCR\\Data\\Citizenship_old\\Annotations\\metadata.jsonl"

from pathlib import Path

# =========================================================
# BASE DIRECTORY
# =========================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# =========================================================
# DATA ROOT
# =========================================================

DATA_DIR = BASE_DIR / "Data"

# =========================================================
# DATASET SELECTION
# =========================================================

DATASET_NAME = "Citizenship_old"

# Examples:
# "Citizenship_old"
# "Citizenship_new"
# "Passport_old"
# "Passport_new"

# =========================================================
# PATHS
# =========================================================

DATASET_DIR = DATA_DIR / DATASET_NAME

INPUT_JSON = (
    DATASET_DIR
    / "Annotations"
    / (
        "old_doc_annotation.json"
        if "old" in DATASET_NAME.lower()
        else "new_doc_annotation.json"
    )
)

IMAGE_DIR = DATASET_DIR / "Docs"

OUTPUT_JSONL = (
    DATASET_DIR
    / "Annotations"
    / "metadata.jsonl"
)

# =========================================================
# DEBUG
# =========================================================

print("BASE_DIR:    ", BASE_DIR)
print("DATASET:     ", DATASET_NAME)
print("INPUT_JSON:  ", INPUT_JSON)
print("IMAGE_DIR:   ", IMAGE_DIR)
print("OUTPUT_JSONL:", OUTPUT_JSONL)


# This script converts the Label Studio JSON annotations into a structured format suitable for training a Donut model.


def load_labelstudio_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fields(annotation):
    fields = {}

    results = annotation["annotations"][0]["result"]

    temp = {}

    for item in results:
        if item["type"] == "rectanglelabels":
            region_id = item["id"]
            label = item["value"]["rectanglelabels"][0]
            temp[region_id] = {"label": label}

        elif item["type"] == "textarea":
            region_id = item["id"]
            text = item["value"]["text"][0]
            if region_id not in temp:
                temp[region_id] = {}
            temp[region_id]["text"] = text

    for region_id, value in temp.items():
        if "label" in value and "text" in value:
            fields[value["label"]] = value["text"]

    return fields


def normalize_fields(raw):
    def get(key):
        return raw.get(key, "")

    structured = {
        "name": {
            "dev": get("name_dev"),
            "eng": get("name_eng")
        },
        "gender": {
            "dev": get("gender_dev"),
            "eng": get("gender_eng")
        },
        "citizenship_number": {
            "dev": get("citizenship_number_dev") or get("citizenship_number"),
            "eng": get("citizenship_number_eng")
        },
        "dob": {
            "bs": {
                "year": get("dob_year_bs"),
                "month": get("dob_month_bs"),
                "day": get("dob_day_bs")
            },
            "ad": {
                "year": get("dob_year_ad"),
                "month": get("dob_month_ad"),
                "day": get("dob_day_ad")
            }
        },
        "birth_address": {
            "district": {
                "dev": get("birth_district_dev") or get("district"),
                "eng": get("birth_district_eng")
            },
            "municipality": {
                "dev": get("birth_municipality_dev") or get("municipality"),
                "eng": get("birth_municipality_eng")
            },
            "ward": get("birth_ward_number_dev") or get("ward_number")
        },
        "current_address": {
            "district": {
                "dev": get("current_district_dev"),
                "eng": get("current_district_eng")
            },
            "municipality": {
                "dev": get("current_municipality_dev"),
                "eng": get("current_municipality_eng")
            },
            "ward": get("current_ward_number_dev")
        },
        "parents": {
            "father": {
                "dev": get("father_name_dev") or get("father_name"),
                "eng": get("father_name_eng")
            },
            "mother": {
                "dev": get("mother_name_dev"),
                "eng": get("mother_name_eng")
            }
        },
        "citizenship_type": get("citizenship_type"),
        "issue_date": {
            "bs": get("issue_date_bs") or get("issue_date"),
            "ad": get("issue_date_ad")
        },
        "nationality": get("nationality")
    }

    return structured


def clean_text(text):
    if not text:
        return ""

    text = text.strip()

    # Normalize slashes to dash
    text = re.sub(r"[\\/]", "-", text)

    # Remove multiple dashes
    text = re.sub(r"-+", "-", text)

    return text


def recursive_clean(data):
    if isinstance(data, dict):
        return {k: recursive_clean(v) for k, v in data.items()}
    elif isinstance(data, str):
        return clean_text(data)
    else:
        return data


def main():
    data = load_labelstudio_json(INPUT_JSON)

    os.makedirs("data/processed/images", exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_file:

        for item in data:
            image_path = item["data"]["ocr"]
            image_name = os.path.basename(image_path)

            fields = extract_fields(item)
            structured = normalize_fields(fields)
            structured = recursive_clean(structured)
            gt_json = json.dumps(structured, ensure_ascii=False)

            record = {
                "image": f"images/{image_name}",
                "ground_truth": gt_json
            }

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("✅ Conversion Done!")


if __name__ == "__main__":
    main()
