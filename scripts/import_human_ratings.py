"""
Import and Validate Human Ratings for Judge Evaluation.

Converts data/judge/human_review.csv into canonical data/judge/human_ratings.json.

Strict Validation Rules:
1. Exactly 45 ratings matching sampled example IDs.
2. Zero duplicate or missing IDs.
3. Zero blank ratings (fails closed if ratings not provided).
4. Ratings must be integers in [1, 5].
5. Strictly parses external entries — NEVER generates, infers, or fills default scores.

Usage:
    python scripts/import_human_ratings.py
"""
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    HUMAN_REVIEW_CSV_PATH,
    HUMAN_RATINGS_JSON_PATH,
    JUDGE_SAMPLE_PATH,
    GENERATED_REPLIES_PATH
)

RATING_DIMENSIONS = [
    "groundedness",
    "helpfulness",
    "relevance",
    "brand_alignment",
    "safety",
    "overall"
]

def print_missing_error():
    print(
        "\nERROR:\n"
        "Human ratings have not been provided.\n\n"
        "Run the annotation workflow and populate:\n"
        f"{HUMAN_REVIEW_CSV_PATH}\n",
        file=sys.stderr
    )

def load_expected_ids() -> Set[str]:
    path = JUDGE_SAMPLE_PATH if JUDGE_SAMPLE_PATH.exists() else GENERATED_REPLIES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Sample source file not found at {path}. Run scripts/generate_judge_samples.py first.")
    expected_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                expected_ids.add(json.loads(line)["example_id"])
    return expected_ids

def parse_and_validate_ratings(csv_path: Path = HUMAN_REVIEW_CSV_PATH) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        print_missing_error()
        raise FileNotFoundError(f"Review CSV not found: {csv_path}")

    expected_ids = load_expected_ids()
    expected_count = len(expected_ids)
    if expected_count != 45:
        raise ValueError(f"Expected exactly 45 sample IDs, but found {expected_count} in sample file.")

    rows = []
    seen_ids = set()

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            eid = row.get("example_id", "").strip()
            if not eid:
                raise ValueError(f"Row {row_idx} is missing 'example_id'.")

            if eid in seen_ids:
                raise ValueError(f"Duplicate example_id found: '{eid}' at row {row_idx}.")
            seen_ids.add(eid)

            if eid not in expected_ids:
                raise ValueError(f"Unknown example_id '{eid}' at row {row_idx}; not in expected 45 judge samples.")

            # Validate ratings are filled
            parsed_scores = {}
            for dim in RATING_DIMENSIONS:
                val_raw = row.get(dim, "").strip()
                if not val_raw:
                    print_missing_error()
                    raise ValueError(f"Blank rating for dimension '{dim}' in example '{eid}'.")

                try:
                    # Must be an integer or integer float
                    f_val = float(val_raw)
                    if not f_val.is_integer():
                        raise ValueError(f"Rating for '{dim}' in example '{eid}' must be an integer, got '{val_raw}'.")
                    int_val = int(f_val)
                except ValueError as e:
                    raise ValueError(f"Invalid numeric rating '{val_raw}' for '{dim}' in example '{eid}': {e}")

                if int_val < 1 or int_val > 5:
                    raise ValueError(f"Rating {int_val} for '{dim}' in example '{eid}' is out of range [1, 5].")

                parsed_scores[dim] = int_val

            notes = row.get("notes", "").strip()

            record = {
                "example_id": eid,
                "rater_id": "human_1",
                "rater_type": "human",
                "groundedness": parsed_scores["groundedness"],
                "helpfulness": parsed_scores["helpfulness"],
                "relevance": parsed_scores["relevance"],
                "brand_alignment": parsed_scores["brand_alignment"],
                "safety": parsed_scores["safety"],
                "overall": parsed_scores["overall"],
                "notes": notes
            }
            rows.append(record)

    # Check for missing IDs
    missing_ids = expected_ids - seen_ids
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} expected example IDs from CSV: {sorted(missing_ids)}")

    if len(rows) != 45:
        raise ValueError(f"Expected exactly 45 ratings, parsed {len(rows)}.")

    return rows

def import_human_ratings():
    try:
        print(f"Validating human ratings in {HUMAN_REVIEW_CSV_PATH}...")
        records = parse_and_validate_ratings(HUMAN_REVIEW_CSV_PATH)
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    HUMAN_RATINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HUMAN_RATINGS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Successfully validated and imported {len(records)} human ratings to {HUMAN_RATINGS_JSON_PATH}")
    print("All ratings verified as genuine integer scores in [1, 5] with rater_type = 'human'.")

if __name__ == "__main__":
    import_human_ratings()
