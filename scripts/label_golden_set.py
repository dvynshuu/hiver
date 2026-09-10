"""
Local Human Annotation Tool for Golden Evaluation Set.
Allows a human annotator to inspect each candidate conversation, label intent,
determine human escalation with explicit reason, score difficulty, and record notes.

Saves incrementally after every single example to ensure progress is never lost.
Raw annotations are saved to data/golden/manual_annotations.jsonl.

Usage:
    python scripts/label_golden_set.py          # Interactive labeling UI
    python scripts/label_golden_set.py --stats  # View distribution of annotations
    python scripts/label_golden_set.py --verify # Verify all 200 items are labeled
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import DATA_DIR, INTENT_TAXONOMY, INTENT_NAMES

GOLDEN_DIR = DATA_DIR / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_POOL_PATH = GOLDEN_DIR / "candidates_pool.jsonl"
MANUAL_ANNOTATIONS_PATH = GOLDEN_DIR / "manual_annotations.jsonl"
SECOND_ANNOTATOR_PATH = GOLDEN_DIR / "second_annotator_sample.jsonl"

ANNOTATOR_ID = "human_single_annotator"

def load_candidates() -> List[Dict[str, Any]]:
    if not CANDIDATES_POOL_PATH.exists():
        from scripts.sample_candidates import sample_candidates
        sample_candidates()
    
    candidates = []
    with open(CANDIDATES_POOL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    return candidates

def load_existing_annotations() -> Dict[str, Dict[str, Any]]:
    if not MANUAL_ANNOTATIONS_PATH.exists():
        return {}
    annotations = {}
    with open(MANUAL_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                annotations[item["example_id"]] = item
    return annotations

def save_single_annotation(annotation: Dict[str, Any]):
    existing = load_existing_annotations()
    existing[annotation["example_id"]] = annotation

    # Deterministic write sorted by example_id
    sorted_items = sorted(existing.values(), key=lambda x: x["example_id"])
    temp_path = MANUAL_ANNOTATIONS_PATH.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for item in sorted_items:
            f.write(json.dumps(item) + "\n")
    temp_path.replace(MANUAL_ANNOTATIONS_PATH)

def print_example_ui(idx: int, total: int, cand: Dict[str, Any], existing: Optional[Dict[str, Any]]):
    print("\n" + "=" * 50)
    print(f"Example {idx + 1} / {total} [{cand['example_id']}]")
    print("=" * 50)
    print("\nConversation context:")
    print(cand.get("context", "N/A"))
    print("\nCustomer message:")
    print(f"\"{cand.get('customer_text', '')}\"")
    
    sug = cand.get("machine_suggestion", {})
    if sug:
        print(f"\n[Machine suggestion (non-binding): {sug.get('intent')} | Escalate: {sug.get('expected_escalation')}]")

    if existing:
        print(f"[CURRENT SAVED ANNOTATION: {existing.get('intent')} | Escalate: {existing.get('expected_escalation')} | Diff: {existing.get('difficulty')}]")

    print("\nIntent:")
    for i, name in enumerate(INTENT_NAMES, 1):
        print(f"{i}. {name}")

def run_labeler():
    candidates = load_candidates()
    total = len(candidates)
    existing_map = load_existing_annotations()

    print(f"Loaded {total} candidates. Already labeled: {len(existing_map)} / {total}")
    print("Commands at any prompt: 'q' to quit, 'p' for previous, 'skip' to skip.")

    # Find first unannotated index
    start_idx = 0
    for i, c in enumerate(candidates):
        if c["example_id"] not in existing_map:
            start_idx = i
            break

    idx = start_idx
    while 0 <= idx < total:
        cand = candidates[idx]
        eid = cand["example_id"]
        curr_anno = existing_map.get(eid)

        print_example_ui(idx, total, cand, curr_anno)

        # 1. Choose intent
        sug_intent = cand.get("machine_suggestion", {}).get("intent", "other")
        def_intent_idx = INTENT_NAMES.index(curr_anno["intent"]) + 1 if curr_anno else (INTENT_NAMES.index(sug_intent) + 1 if sug_intent in INTENT_NAMES else 8)

        intent_input = input(f"\nChoose intent [1-8, default {def_intent_idx}]: ").strip().lower()
        if intent_input == "q":
            print("Exiting labeling tool. Progress preserved.")
            break
        elif intent_input == "p":
            idx = max(0, idx - 1)
            continue
        elif intent_input == "skip":
            idx += 1
            continue
        elif intent_input == "":
            selected_intent = INTENT_NAMES[def_intent_idx - 1]
        elif intent_input in [str(i) for i in range(1, len(INTENT_NAMES) + 1)]:
            selected_intent = INTENT_NAMES[int(intent_input) - 1]
        elif intent_input in INTENT_NAMES:
            selected_intent = intent_input
        else:
            print("Invalid intent option. Retrying this example.")
            continue

        # 2. Escalate? [y/n]
        sug_esc = cand.get("machine_suggestion", {}).get("expected_escalation", False)
        def_esc = "y" if (curr_anno["expected_escalation"] if curr_anno else sug_esc) else "n"
        esc_input = input(f"Escalate?\n[y/n, default {def_esc}]: ").strip().lower()

        if esc_input == "q":
            break
        elif esc_input == "":
            esc_val = (def_esc == "y")
        elif esc_input in ["y", "yes"]:
            esc_val = True
        elif esc_input in ["n", "no"]:
            esc_val = False
        else:
            print("Invalid escalation option (must be y/n). Retrying.")
            continue

        # 3. Escalation reason
        esc_reason = ""
        if esc_val:
            sug_reason = cand.get("machine_suggestion", {}).get("escalation_reason", "")
            def_reason = curr_anno.get("escalation_reason", sug_reason) if curr_anno else sug_reason
            reason_input = input(f"\nIf yes, escalation reason:\n[default: {def_reason}]: ").strip()
            esc_reason = reason_input if reason_input else def_reason
        else:
            esc_reason = "Standard support inquiry suitable for automated guidance"

        # 4. Difficulty [easy / medium / hard]
        sug_diff = cand.get("machine_suggestion", {}).get("difficulty", "medium")
        def_diff = curr_anno.get("difficulty", sug_diff) if curr_anno else sug_diff
        diff_input = input(f"\nDifficulty:\n[easy / medium / hard, default {def_diff}]: ").strip().lower()
        if diff_input in ["easy", "medium", "hard"]:
            difficulty = diff_input
        elif diff_input == "":
            difficulty = def_diff
        else:
            difficulty = "medium"

        # 5. Notes
        def_notes = curr_anno.get("notes", "") if curr_anno else ""
        notes_input = input(f"\nNotes:\n[default: '{def_notes}']: ").strip()
        notes = notes_input if notes_input else def_notes

        # Construct verified human annotation record
        record = {
            "example_id": cand["example_id"],
            "conversation_id": cand["conversation_id"],
            "customer_tweet_id": cand["customer_tweet_id"],
            "customer_text": cand["customer_text"],
            "context": cand["context"],
            "source": cand["source"],
            "split": "held_out",
            "intent": selected_intent,
            "expected_escalation": esc_val,
            "escalation_reason": esc_reason,
            "difficulty": difficulty,
            "annotator": "human",
            "annotator_type": ANNOTATOR_ID,
            "notes": notes,
            "ground_truth_reply": cand.get("support_reply", "")
        }

        save_single_annotation(record)
        existing_map[eid] = record
        print(f"Saved example {cand['example_id']}. ({len(existing_map)} / {total} completed)")
        idx += 1

    print("\nAnnotation session complete. Saved to:", MANUAL_ANNOTATIONS_PATH)

def print_stats():
    annotations = load_existing_annotations()
    print(f"Total annotations saved: {len(annotations)}")
    if not annotations:
        return

    intent_counts = {}
    esc_counts = {}
    diff_counts = {}
    source_counts = {}

    for a in annotations.values():
        intent_counts[a["intent"]] = intent_counts.get(a["intent"], 0) + 1
        esc_counts[a["expected_escalation"]] = esc_counts.get(a["expected_escalation"], 0) + 1
        diff_counts[a["difficulty"]] = diff_counts.get(a["difficulty"], 0) + 1
        source_counts[a["source"]] = source_counts.get(a["source"], 0) + 1

    print("\nIntent Distribution:")
    for k in INTENT_NAMES:
        print(f"  {k:<18}: {intent_counts.get(k, 0)}")

    print("\nEscalation Distribution:")
    for k, v in esc_counts.items():
        print(f"  Escalate={k}: {v}")

    print("\nDifficulty Distribution:")
    for k, v in diff_counts.items():
        print(f"  {k}: {v}")

    print("\nSource Distribution:")
    for k, v in source_counts.items():
        print(f"  {k}: {v}")

def verify_dataset() -> bool:
    candidates = load_candidates()
    annotations = load_existing_annotations()

    if len(annotations) != len(candidates):
        print(f"VERIFY FAILED: {len(annotations)} annotations found, expected {len(candidates)}")
        return False

    # Check required provenance fields
    required_fields = [
        "example_id", "conversation_id", "customer_tweet_id", "customer_text",
        "context", "source", "split", "intent", "expected_escalation",
        "difficulty", "annotator", "annotator_type"
    ]

    intent_counts = {name: 0 for name in INTENT_NAMES}

    for eid, item in annotations.items():
        for field in required_fields:
            if field not in item:
                print(f"VERIFY FAILED: Item {eid} missing required field '{field}'")
                return False
        
        intent = item["intent"]
        if intent not in INTENT_NAMES:
            print(f"VERIFY FAILED: Item {eid} has invalid intent '{intent}'")
            return False
        intent_counts[intent] += 1

        if item["annotator_type"] != ANNOTATOR_ID:
            print(f"VERIFY FAILED: Item {eid} has dishonest annotator identity '{item['annotator_type']}'")
            return False

    # Check intent coverage
    zero_coverage = [name for name, count in intent_counts.items() if count == 0]
    if zero_coverage:
        print(f"VERIFY FAILED: Intents with 0 coverage: {zero_coverage}")
        return False

    low_coverage = [name for name, count in intent_counts.items() if count < 5]
    if low_coverage:
        print(f"VERIFY WARNING: Intents with <5 coverage: {low_coverage}")

    print("VERIFY PASSED: All 200 items contain valid provenance, single annotator honesty, and intent coverage.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Set Human Annotation Tool")
    parser.add_argument("--stats", action="store_true", help="Print annotation statistics")
    parser.add_argument("--verify", action="store_true", help="Verify dataset completeness")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.verify:
        success = verify_dataset()
        sys.exit(0 if success else 1)
    else:
        run_labeler()
