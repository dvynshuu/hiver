"""
Compile the verified, human-annotated Golden Evaluation Set for @AppleSupport.
Strictly compiles manual annotations from data/golden/manual_annotations.jsonl,
enforcing explicit provenance, zero leakage, and intent coverage.

DOES NOT generate heuristic labels as ground truth.
Workflow:
candidate generation -> manual annotation -> golden dataset -> evaluation.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    DATA_DIR,
    RETRIEVAL_CORPUS_JSONL_PATH,
    GOLDEN_EVAL_PATH,
    INTENT_NAMES
)
from src.data_pipeline import check_evaluation_leakage, format_leakage_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DIR = DATA_DIR / "golden"
MANUAL_ANNOTATIONS_PATH = GOLDEN_DIR / "manual_annotations.jsonl"
CANDIDATES_POOL_PATH = GOLDEN_DIR / "candidates_pool.jsonl"

def compile_golden_eval_dataset():
    logger.info("Compiling verified Golden Evaluation Set from human annotations...")

    if not MANUAL_ANNOTATIONS_PATH.exists():
        logger.info(f"Manual annotations not found at {MANUAL_ANNOTATIONS_PATH}. Initializing candidate sampling...")
        from scripts.sample_candidates import sample_candidates
        sample_candidates()
        raise FileNotFoundError(
            f"Please run 'python scripts/label_golden_set.py' to complete manual annotations at {MANUAL_ANNOTATIONS_PATH}"
        )

    eval_items = []
    with open(MANUAL_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))

    assert len(eval_items) == 200, f"Expected exactly 200 golden examples, got {len(eval_items)}"

    # Validate provenance and human ground truth
    required_provenance = [
        "example_id", "conversation_id", "customer_tweet_id", "customer_text",
        "context", "source", "split", "intent", "expected_escalation",
        "difficulty", "annotator", "annotator_type"
    ]

    intent_counts = {k: 0 for k in INTENT_NAMES}
    for item in eval_items:
        for key in required_provenance:
            if key not in item:
                raise ValueError(f"Golden item {item.get('example_id')} missing required provenance field: '{key}'")

        if item["annotator_type"] != "human_single_annotator":
            raise ValueError(f"Dishonest or invalid annotator identity in item {item.get('example_id')}: {item.get('annotator_type')}")

        # Ensure compatibility fields for evaluator
        item["customer_message"] = item["customer_text"]
        item["ground_truth_intent"] = item["intent"]
        item["ground_truth_escalation"] = "escalate" if item["expected_escalation"] else "auto_handle"
        intent_counts[item["intent"]] += 1

    # Validate intent coverage: every intent must have at least 5 examples
    for intent_name, count in intent_counts.items():
        if count < 5:
            raise ValueError(f"Intent '{intent_name}' has insufficient coverage ({count} < 5 examples)!")

    # Write to final GOLDEN_EVAL_PATH
    with open(GOLDEN_EVAL_PATH, "w", encoding="utf-8") as f:
        for it in eval_items:
            f.write(json.dumps(it) + "\n")

    logger.info(f"Saved {len(eval_items)} verified golden evaluation examples to {GOLDEN_EVAL_PATH}")

    # Automated leakage check against retrieval corpus
    logger.info("Executing automated leakage detection against retrieval corpus...")
    retrieval_corpus = []
    if RETRIEVAL_CORPUS_JSONL_PATH.exists():
        with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    retrieval_corpus.append(json.loads(line))

    leakage_result = check_evaluation_leakage(retrieval_corpus, eval_items)
    print("\n" + format_leakage_report(leakage_result) + "\n")

    if leakage_result["status"] != "PASS":
        raise ValueError(f"CRITICAL: Evaluation leakage detected: {leakage_result}")

    # Summary report
    esc_counts = {}
    source_counts = {}
    for it in eval_items:
        esc_counts[it["ground_truth_escalation"]] = esc_counts.get(it["ground_truth_escalation"], 0) + 1
        source_counts[it["source"]] = source_counts.get(it["source"], 0) + 1

    print("==================================================")
    print("GOLDEN EVALUATION SET COMPILED")
    print("==================================================")
    print(f"Total Examples:      {len(eval_items)}")
    print(f"Human-Labelled:      {len(eval_items)} (annotator: human_single_annotator)")
    print(f"Sources:             {source_counts}")
    print(f"Escalation Split:    {esc_counts}")
    print("Intent Distribution:")
    for k in INTENT_NAMES:
        print(f"  {k:<18}: {intent_counts.get(k, 0)}")
    print("==================================================")

if __name__ == "__main__":
    compile_golden_eval_dataset()
