"""
postprocess.py
───────────────
Post-processing for Donut model output.

Steps:
  1. Parse raw XML output → nested dict
  2. Regex validate field formats (citizenship number, BS dates)
  3. OCR correction (common Devanagari substitution errors)
  4. Field-level confidence scoring
  5. Return clean structured JSON

Used by api.py at inference time.
Can also be run standalone to test on a raw output string.

Run from CustomOCR/ root:
  python postprocess.py
"""

import re
import json
from difflib import SequenceMatcher

# ─── DIGIT MAPS ───────────────────────────────────────────────────────────────

DEV_TO_ENG = str.maketrans("०१२३४५६७८९", "0123456789")
ENG_TO_DEV = str.maketrans("0123456789", "०१२३४५६७८९")

def to_eng_digits(s): return str(s).translate(DEV_TO_ENG)
def to_dev_digits(s): return str(s).translate(ENG_TO_DEV)

# ─── COMMON OCR CORRECTION MAPS ───────────────────────────────────────────────
# Common Devanagari character confusions made by Donut
# Format: {wrong: correct}

DEV_SUBSTITUTIONS = {
    # Visually similar consonants
    "ब": "व",   # ba/va confusion (context-dependent — applied carefully)
    "ण": "न",   # retroflex na vs dental na (less common in names)
    # Matras
    "े": "े",   # normalize composed vs decomposed (Unicode normalization handles this)
    # Numerals
    "७": "७",   # normalize
}

# Common English OCR errors in Nepali transliteration
ENG_SUBSTITUTIONS = {
    "0": "O",   # zero vs letter O in names (context-dependent)
    "1": "I",   # one vs I (context-dependent)
}

# ─── REGEX VALIDATORS ─────────────────────────────────────────────────────────

# New citizenship number: DD-0P-YY-NNNNN (Devanagari)
# e.g. ३२-०१-७८-०४४६४
CITZ_NUM_NEW_DEV = re.compile(
    r"^[०-९]{2}-[०-९]{2}-[०-९]{2}-[०-९]{4,6}$"
)
# English: 32-01-78-04464
CITZ_NUM_NEW_ENG = re.compile(
    r"^\d{2}-\d{2}-\d{2}-\d{4,6}$"
)
# Old citizenship: long number
CITZ_NUM_OLD_DEV = re.compile(r"^[०-९]{7,12}$")
CITZ_NUM_OLD_ENG = re.compile(r"^\d{7,12}$")

# BS date year: 2000-2082
BS_YEAR_DEV = re.compile(r"^[२][०-९]{3}$")   # starts with २ (2xxx)
BS_MONTH_DEV = re.compile(r"^[०-९]{1,2}$")
BS_DAY_DEV   = re.compile(r"^[०-९]{1,2}$")

# Issue date BS: YYYY-MM-DD or YYYY/MM/DD
ISSUE_DATE_BS = re.compile(
    r"^[२][०-९]{3}[-/][०-९]{1,2}[-/][०-९]{1,2}$"
)

# Gender Devanagari
GENDER_DEV_VALID = {"महिला", "पुरुष", "पु", "म"}
GENDER_ENG_VALID = {"Male", "Female", "MALE", "FEMALE"}

# Citizenship types
CITZ_TYPE_VALID = {"वंशज", "जन्म", "अंगिकृत", "बैवाहिक अंगिकृत"}

# Nationality
NATIONALITY_VALID = {"नेपाली"}

# ─── FIELD-LEVEL CONFIDENCE ───────────────────────────────────────────────────

def field_confidence(field_name, value, validation_result):
    """
    Compute a confidence score [0.0, 1.0] for a field value.

    Factors:
    - validation_result: bool (passes regex/enum check)
    - value length: empty = 0.0
    - character set: mixed scripts reduce confidence
    """
    if not value or not value.strip():
        return 0.0

    # Base score from validation
    base = 1.0 if validation_result else 0.5

    # Penalize very short values for fields that should be longer
    long_fields = {"name.dev", "name.eng", "father_name.dev",
                   "birth_address.municipality.dev"}
    if field_name in long_fields and len(value.strip()) < 3:
        base *= 0.5

    # Penalize unexpected script mixing
    has_dev = any(0x0900 <= ord(c) <= 0x097F for c in value)
    has_eng = any(c.isascii() and c.isalpha() for c in value)
    if has_dev and has_eng:
        # Mixed script — OK for fields like citizenship_number but not for names
        name_fields = {"name.dev", "parents.father.dev", "parents.mother.dev"}
        if field_name in name_fields:
            base *= 0.7

    return round(min(1.0, max(0.0, base)), 3)


# ─── VALIDATORS ───────────────────────────────────────────────────────────────

def validate_citizenship_number(dev_val, eng_val):
    """Validate citizenship number in both scripts. Returns (is_valid, corrected_dev, corrected_eng)."""
    dev_clean = dev_val.strip()
    eng_clean = eng_val.strip()

    dev_valid = (CITZ_NUM_NEW_DEV.match(dev_clean) is not None or
                 CITZ_NUM_OLD_DEV.match(dev_clean) is not None)
    eng_valid = (CITZ_NUM_NEW_ENG.match(eng_clean) is not None or
                 CITZ_NUM_OLD_ENG.match(eng_clean) is not None or
                 not eng_clean)   # empty eng is OK for old docs

    # Cross-check: if one is valid and other empty, derive the other
    if dev_valid and not eng_clean:
        eng_clean = to_eng_digits(dev_clean)
        eng_valid = True
    elif eng_valid and not dev_clean:
        dev_clean = to_dev_digits(eng_clean)
        dev_valid = True

    return dev_valid and eng_valid, dev_clean, eng_clean


def validate_bs_date(year, month, day):
    """Validate BS date components. Returns is_valid bool."""
    year_clean  = year.strip()
    month_clean = month.strip()
    day_clean   = day.strip()

    if not year_clean:
        return False

    year_valid  = BS_YEAR_DEV.match(year_clean) is not None
    month_valid = (BS_MONTH_DEV.match(month_clean) is not None
                   and 1 <= int(to_eng_digits(month_clean) or "0") <= 12)
    day_valid   = (BS_DAY_DEV.match(day_clean) is not None
                   and 1 <= int(to_eng_digits(day_clean) or "0") <= 32)

    return year_valid and month_valid and day_valid


def validate_gender(dev_val, eng_val):
    """Normalize and validate gender."""
    dev_clean = dev_val.strip()
    eng_clean = eng_val.strip()

    # Normalize abbreviations
    if dev_clean in {"पु", "पुरुष"}:
        dev_clean = "पुरुष"
        if not eng_clean:
            eng_clean = "Male"
    elif dev_clean in {"म", "महिला"}:
        dev_clean = "महिला"
        if not eng_clean:
            eng_clean = "Female"

    dev_valid = dev_clean in GENDER_DEV_VALID or not dev_clean
    eng_valid = eng_clean in GENDER_ENG_VALID or not eng_clean

    return dev_valid and eng_valid, dev_clean, eng_clean


def validate_issue_date(bs_val):
    """Validate BS issue date string."""
    val = bs_val.strip()
    if not val:
        return False, val
    valid = ISSUE_DATE_BS.match(val) is not None
    return valid, val


# ─── XML PARSER ───────────────────────────────────────────────────────────────

def parse_xml_output(xml_str):
    """
    Parse Donut XML output → nested dict.
    Handles malformed/incomplete XML gracefully.
    """
    result  = {}
    pattern = re.compile(r"<s_([^>]+?)>(.*?)</s_\1>", re.DOTALL)

    for tag, value in pattern.findall(xml_str):
        if tag in ("nepali_citizenship",):
            continue
        # Convert underscore-separated tag back to nested structure
        parts  = tag.split("_", 1)
        if len(parts) == 2:
            outer, inner = parts
            result.setdefault(outer, {})[inner] = value.strip()
        else:
            result[tag] = value.strip()

    return result


def build_nested(flat_result):
    """
    Rebuild standard nested output structure from flat parse result.
    Ensures all expected keys exist even if model didn't output them.
    """
    def get(d, *keys, default=""):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d if isinstance(d, str) else default

    r = flat_result  # shorthand

    return {
        "name": {
            "dev": get(r, "name", "dev"),
            "eng": get(r, "name", "eng"),
        },
        "gender": {
            "dev": get(r, "gender", "dev"),
            "eng": get(r, "gender", "eng"),
        },
        "nationality": get(r, "nationality") if isinstance(r.get("nationality"), str)
                       else get(r, "nationality", "dev", default="नेपाली"),
        "citizenship_number": {
            "dev": get(r, "citizenship", "number_dev") or get(r, "citizenship_number", "dev"),
            "eng": get(r, "citizenship", "number_eng") or get(r, "citizenship_number", "eng"),
        },
        "citizenship_type": get(r, "citizenship", "type") or get(r, "citizenship_type"),
        "dob": {
            "bs": {
                "year":  get(r, "dob", "bs_year")  or get(r, "dob", "year"),
                "month": get(r, "dob", "bs_month") or get(r, "dob", "month"),
                "day":   get(r, "dob", "bs_day")   or get(r, "dob", "day"),
            },
            "ad": {
                "year":  get(r, "dob", "ad_year"),
                "month": get(r, "dob", "ad_month"),
                "day":   get(r, "dob", "ad_day"),
            },
        },
        "birth_address": {
            "district":     {"dev": get(r, "birth", "address_district_dev"),
                             "eng": get(r, "birth", "address_district_eng")},
            "municipality": {"dev": get(r, "birth", "address_municipality_dev"),
                             "eng": get(r, "birth", "address_municipality_eng")},
            "ward":         get(r, "birth", "address_ward"),
        },
        "current_address": {
            "district":     {"dev": get(r, "current", "address_district_dev"),
                             "eng": get(r, "current", "address_district_eng")},
            "municipality": {"dev": get(r, "current", "address_municipality_dev"),
                             "eng": get(r, "current", "address_municipality_eng")},
            "ward":         get(r, "current", "address_ward"),
        },
        "parents": {
            "father": {"dev": get(r, "parents", "father_dev"),
                       "eng": get(r, "parents", "father_eng")},
            "mother": {"dev": get(r, "parents", "mother_dev"),
                       "eng": get(r, "parents", "mother_eng")},
        },
        "issue_date": {
            "bs": get(r, "issue", "date_bs") or get(r, "issue_date", "bs"),
            "ad": get(r, "issue", "date_ad") or get(r, "issue_date", "ad"),
        },
    }


# ─── MAIN POSTPROCESSOR ───────────────────────────────────────────────────────

def postprocess(raw_xml_output):
    """
    Full post-processing pipeline.

    Input:  raw XML string from Donut model
    Output: {
        "data":       nested JSON dict (cleaned, validated),
        "confidence": {field: score},
        "warnings":   [list of validation warnings],
        "valid":      bool (all critical fields passed validation)
    }
    """
    warnings    = []
    confidence  = {}

    # Step 1: Parse XML
    flat_result = parse_xml_output(raw_xml_output)
    nested      = build_nested(flat_result)

    # Step 2: Validate + correct citizenship number
    cn_dev = nested["citizenship_number"]["dev"]
    cn_eng = nested["citizenship_number"]["eng"]
    cn_valid, cn_dev_fixed, cn_eng_fixed = validate_citizenship_number(cn_dev, cn_eng)
    nested["citizenship_number"]["dev"] = cn_dev_fixed
    nested["citizenship_number"]["eng"] = cn_eng_fixed
    if not cn_valid:
        warnings.append(f"citizenship_number format invalid: dev='{cn_dev}' eng='{cn_eng}'")
    confidence["citizenship_number.dev"] = field_confidence("citizenship_number.dev", cn_dev_fixed, cn_valid)
    confidence["citizenship_number.eng"] = field_confidence("citizenship_number.eng", cn_eng_fixed, cn_valid)

    # Step 3: Validate BS date
    bs      = nested["dob"]["bs"]
    dob_valid = validate_bs_date(bs.get("year",""), bs.get("month",""), bs.get("day",""))
    if not dob_valid:
        warnings.append(f"dob.bs invalid: {bs}")
    confidence["dob.bs"] = field_confidence("dob.bs", bs.get("year",""), dob_valid)

    # Step 4: Validate + normalize gender
    gen_valid, gen_dev_fixed, gen_eng_fixed = validate_gender(
        nested["gender"]["dev"], nested["gender"]["eng"]
    )
    nested["gender"]["dev"] = gen_dev_fixed
    nested["gender"]["eng"] = gen_eng_fixed
    if not gen_valid:
        warnings.append(f"gender invalid: dev='{nested['gender']['dev']}'")
    confidence["gender.dev"] = field_confidence("gender.dev", gen_dev_fixed, gen_valid)

    # Step 5: Validate issue date
    issue_valid, issue_fixed = validate_issue_date(nested["issue_date"]["bs"])
    nested["issue_date"]["bs"] = issue_fixed
    if not issue_valid and issue_fixed:
        warnings.append(f"issue_date.bs format unexpected: '{issue_fixed}'")
    confidence["issue_date.bs"] = field_confidence("issue_date.bs", issue_fixed, issue_valid)

    # Step 6: Validate citizenship type
    ctype       = nested.get("citizenship_type", "")
    ctype_valid = ctype in CITZ_TYPE_VALID or not ctype
    if not ctype_valid:
        warnings.append(f"citizenship_type unexpected: '{ctype}'")
    confidence["citizenship_type"] = field_confidence("citizenship_type", ctype, ctype_valid)

    # Step 7: Simple confidence for name fields
    name_dev = nested["name"]["dev"]
    confidence["name.dev"] = field_confidence("name.dev", name_dev, bool(name_dev))
    confidence["name.eng"] = field_confidence("name.eng", nested["name"]["eng"],
                                               bool(nested["name"]["eng"]))

    # Step 8: Normalize nationality
    if not nested.get("nationality"):
        nested["nationality"] = "नेपाली"
    confidence["nationality"] = 1.0

    # Overall validity: critical fields must pass
    critical_valid = (
        bool(name_dev) and
        cn_valid and
        dob_valid
    )

    # Overall confidence: mean of all field confidences
    avg_confidence = round(
        sum(confidence.values()) / max(1, len(confidence)), 3
    )

    return {
        "data":            nested,
        "confidence":      confidence,
        "avg_confidence":  avg_confidence,
        "warnings":        warnings,
        "valid":           critical_valid,
    }


# ─── STANDALONE TEST ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a sample raw XML output
    sample_xml = (
        "<s_nepali_citizenship>"
        "<s_name_dev>सृजना रौनीयार</s_name_dev>"
        "<s_name_eng>SRIJANA RAUNIYAR</s_name_eng>"
        "<s_gender_dev>महिला</s_gender_dev>"
        "<s_gender_eng>Female</s_gender_eng>"
        "<s_nationality>नेपाली</s_nationality>"
        "<s_citizenship_number_dev>३२-०१-७८-०४४६४</s_citizenship_number_dev>"
        "<s_citizenship_number_eng>32-01-78-04464</s_citizenship_number_eng>"
        "<s_citizenship_type>वंशज</s_citizenship_type>"
        "<s_dob_bs_year>२०६०</s_dob_bs_year>"
        "<s_dob_bs_month>०४</s_dob_bs_month>"
        "<s_dob_bs_day>१९</s_dob_bs_day>"
        "<s_issue_date_bs>२०७८-०५-१६</s_issue_date_bs>"
        "</s_nepali_citizenship>"
    )

    result = postprocess(sample_xml)

    print("\nPostprocess Test")
    print("─" * 50)
    print(f"Valid:          {result['valid']}")
    print(f"Avg confidence: {result['avg_confidence']}")
    print(f"Warnings:       {result['warnings']}")
    print(f"\nExtracted data:")
    print(json.dumps(result["data"], ensure_ascii=False, indent=2))
    print(f"\nField confidence:")
    for f, c in result["confidence"].items():
        print(f"  {f:<35} {c:.3f}")