"""
Interactive Human Labeling Tool for Golden Evaluation Set.
Allows manual inspection, labeling, and editing of golden examples with incremental auto-saving.
Run:
    python scripts/label_golden_set.py
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GOLDEN_EVAL_PATH, INTENT_TAXONOMY, INTENT_NAMES

def print_banner():
    print("=" * 70)
    print("       HIVER SUPPORT AGENT: GOLDEN EVALUATION LABELING TOOL")
    print("=" * 70)
    print("Commands:")
    print("  [1-8]     Select intent index")
    print("  [e/a]     Escalate (e) or Auto-Handle (a)")
    print("  [n]       Next item (keep current labels)")
    print("  [p]       Previous item")
    print("  [q]       Save and quit")
    print("=" * 70)

def load_dataset() -> List[Dict[str, Any]]:
    if not GOLDEN_EVAL_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found at {GOLDEN_EVAL_PATH}")
    items = []
    with open(GOLDEN_EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def save_dataset(items: List[Dict[str, Any]]):
    with open(GOLDEN_EVAL_PATH, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

def run_labeler():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print_banner()
    items = load_dataset()
    total = len(items)
    idx = 0

    while 0 <= idx < total:
        item = items[idx]
        print(f"\n--- [{idx + 1} / {total}] Example: {item.get('example_id', 'unknown')} (Source: {item.get('source')}) ---")
        print(f"Context:          {item.get('context', 'None')}")
        print(f"Customer Message: \"{item.get('customer_text', '')}\"")
        print(f"Reference Reply:  \"{item.get('ground_truth_reply', '')[:100]}...\"")
        print(f"\nCurrent Intent:   {item.get('ground_truth_intent')} ({item.get('difficulty', 'medium')})")
        print(f"Escalation:       {item.get('ground_truth_escalation')} | Trigger: {item.get('escalation_trigger', 'none')}")
        print(f"Reason:           {item.get('escalation_reason', '')}")
        print("-" * 70)
        print("Available Intents:")
        for i, name in enumerate(INTENT_NAMES, 1):
            print(f"  {i}. {name:<18} ({INTENT_TAXONOMY[name]['name']})")
        
        choice = input("\nAction [Enter=keep/next, 1-8=change intent, e=escalate, a=auto, p=prev, q=quit]: ").strip().lower()

        if choice == "q":
            save_dataset(items)
            print(f"\nProgress saved to {GOLDEN_EVAL_PATH}. Exiting.")
            break
        elif choice == "p":
            idx = max(0, idx - 1)
            continue
        elif choice in [str(i) for i in range(1, len(INTENT_NAMES) + 1)]:
            new_intent = INTENT_NAMES[int(choice) - 1]
            item["ground_truth_intent"] = new_intent
            item["intent"] = new_intent
            print(f"-> Intent updated to: {new_intent}")
            esc_in = input("Escalate? (y/n, default keep): ").strip().lower()
            if esc_in == "y":
                item["expected_escalation"] = True
                item["ground_truth_escalation"] = "escalate"
                reason_in = input("Escalation reason: ").strip()
                if reason_in:
                    item["escalation_reason"] = reason_in
            elif esc_in == "n":
                item["expected_escalation"] = False
                item["ground_truth_escalation"] = "auto_handle"

            save_dataset(items)
            print("Saved incrementally.")
            idx += 1
        elif choice == "e":
            item["expected_escalation"] = True
            item["ground_truth_escalation"] = "escalate"
            reason_in = input("Escalation reason: ").strip()
            if reason_in:
                item["escalation_reason"] = reason_in
            save_dataset(items)
            print("Saved as ESCALATE.")
            idx += 1
        elif choice == "a":
            item["expected_escalation"] = False
            item["ground_truth_escalation"] = "auto_handle"
            save_dataset(items)
            print("Saved as AUTO_HANDLE.")
            idx += 1
        else:
            idx += 1

    print("\nLabeling complete or exited. Dataset intact.")

if __name__ == "__main__":
    run_labeler()
