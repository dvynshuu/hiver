"""
Export Human Review Form for Judge Validation.

Reads generated agent replies and outputs a clean review CSV for manual human annotation.
Rating columns (groundedness, helpfulness, relevance, brand_alignment, safety, overall, notes)
are strictly left blank.

Usage:
    python scripts/export_human_review.py
    python scripts/export_human_review.py --force
"""
import sys
import json
import csv
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import GENERATED_REPLIES_PATH, HUMAN_REVIEW_CSV_PATH

FIELDNAMES = [
    "example_id",
    "customer_message",
    "conversation_context",
    "retrieved_evidence",
    "agent_reply",
    "groundedness",
    "helpfulness",
    "relevance",
    "brand_alignment",
    "safety",
    "overall",
    "notes"
]

def format_evidence_text(evidence_list):
    if not evidence_list:
        return "No historical evidence retrieved."
    lines = []
    for ev in evidence_list:
        pid = ev.get("pair_id", "N/A")
        q = (ev.get("query") or "").replace("\n", " ")
        r = (ev.get("response") or "").replace("\n", " ")
        lines.append(f"[{pid}] Cust: {q} | Agent: {r}")
    return "\n".join(lines)

def export_human_review(force: bool = False):
    if not GENERATED_REPLIES_PATH.exists():
        raise FileNotFoundError(
            f"Generated replies not found at {GENERATED_REPLIES_PATH}. "
            f"Run scripts/generate_judge_samples.py first."
        )

    if HUMAN_REVIEW_CSV_PATH.exists() and not force:
        # Check if already populated with human annotations
        with open(HUMAN_REVIEW_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            has_entries = any(row.get("overall", "").strip() for row in reader)
        if has_entries:
            print(f"Notice: {HUMAN_REVIEW_CSV_PATH} already contains human ratings.")
            print("To re-export a blank review template, rerun with --force.")
            return

    records = []
    with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    HUMAN_REVIEW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(HUMAN_REVIEW_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for rec in records:
            ev_text = format_evidence_text(rec.get("retrieved_evidence", []))
            writer.writerow({
                "example_id": rec["example_id"],
                "customer_message": rec.get("customer_text", ""),
                "conversation_context": rec.get("context", ""),
                "retrieved_evidence": ev_text,
                "agent_reply": rec.get("agent_reply") or "[ESCALATED - NO AUTOMATED REPLY DISPATCHED]",
                "groundedness": "",
                "helpfulness": "",
                "relevance": "",
                "brand_alignment": "",
                "safety": "",
                "overall": "",
                "notes": ""
            })

    print(f"Successfully exported {len(records)} review cases to {HUMAN_REVIEW_CSV_PATH}")
    print("All rating columns are strictly blank. Refer to data/judge/HUMAN_RATING_GUIDE.md for the rubric.")

def main():
    parser = argparse.ArgumentParser(description="Export human review form for agent replies.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing review CSV even if ratings exist.")
    args = parser.parse_args()
    export_human_review(force=args.force)

if __name__ == "__main__":
    main()
