"""
Canonical Evaluation Leakage Gate.
Audits retrieval corpus against evaluation sets for:
1. Exact customer-text overlap
2. Normalized customer-text overlap
3. Conversation ID overlap

Outputs:
Text overlap: 0
Conversation overlap: 0
Status: PASS
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, RETRIEVAL_CORPUS_JSONL_PATH, GOLDEN_EVAL_PATH
from src.data_pipeline import check_evaluation_leakage

def run_leakage_check() -> bool:
    if not RETRIEVAL_CORPUS_JSONL_PATH.exists():
        print("FAIL: Retrieval corpus not found.")
        sys.exit(1)

    golden_path = GOLDEN_EVAL_PATH
    if not golden_path.exists():
        alt_path = DATA_DIR / "golden" / "golden_eval.jsonl"
        if alt_path.exists():
            golden_path = alt_path
        else:
            print("FAIL: Golden evaluation set not found.")
            sys.exit(1)

    retrieval_corpus = []
    with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                retrieval_corpus.append(json.loads(line))

    eval_set = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_set.append(json.loads(line))

    result = check_evaluation_leakage(retrieval_corpus, eval_set)

    # Print the exact output format specified in Phase 1
    total_text_overlap = result["exact_text_overlap"] + result["normalized_overlap"]
    conv_overlap = result["conversation_overlap"]
    status = result["status"]

    print(f"Text overlap: {total_text_overlap}")
    print(f"Conversation overlap: {conv_overlap}")
    print(f"Status: {status}")

    if status != "PASS":
        sys.exit(1)
    return True

if __name__ == "__main__":
    run_leakage_check()
