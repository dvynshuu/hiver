"""
Canonical Candidate Sampler for Golden Evaluation Set.
Samples 180 real held-out customer support conversations from TWCS using:
- Stratification across all 8 taxonomy intents (guaranteeing >= 5 per class)
- Rare cases, ambiguous cases, and varying conversation lengths
- Deterministic random seed (default: 42)
Appends 20 hard/adversarial edge cases (total: 200 candidates).

Outputs:
data/golden/candidates.jsonl
"""
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import DATA_DIR, SEED, INTENT_NAMES

GOLDEN_DIR = DATA_DIR / "golden"
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_PATH = GOLDEN_DIR / "candidates.jsonl"
HELD_OUT_PATH = DATA_DIR / "held_out_pool.jsonl"

from scripts.build_golden_eval_set import ADVERSARIAL_CASES, classify_text_heuristically, determine_escalation_heuristically

def sample_candidates(seed: int = SEED, count: int = 180):
    print(f"Loading held-out pool from {HELD_OUT_PATH}...")
    pool_items = []
    with open(HELD_OUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pool_items.append(json.loads(line))

    # Clean English filtering
    clean_items = []
    for it in pool_items:
        txt = it.get("customer_text", "")
        if len(txt) < 22:
            continue
        ascii_ratio = sum(1 for c in txt if ord(c) < 128) / len(txt)
        if ascii_ratio < 0.85:
            continue
        clean_items.append(it)

    print(f"Clean held-out pool candidates available: {len(clean_items)}")

    # Specific pre-screened product inquiry pair_ids to guarantee product_inquiry representation
    known_pi_pids = {
        "apple_02473", "apple_00898", "apple_03124", "apple_00617",
        "apple_02135", "apple_02239", "apple_00174", "apple_01713"
    }

    buckets = {name: [] for name in INTENT_NAMES}
    for it in clean_items:
        pid = it.get("pair_id")
        txt = it.get("customer_text", "")
        if pid in known_pi_pids:
            cand_intent = "product_inquiry"
        else:
            cand_intent = classify_text_heuristically(txt)
        buckets[cand_intent].append(it)

    print("Candidate intent buckets distribution in pool:")
    for k, v in buckets.items():
        print(f"  {k:<18}: {len(v)}")

    rng = random.Random(seed)
    selected_real = []
    selected_pids = set()

    # Stratified guarantee: minimum 6 per bucket
    min_per_bucket = 6
    for name in INTENT_NAMES:
        b_items = list(buckets[name])
        rng.shuffle(b_items)
        pick_count = min(min_per_bucket, len(b_items))
        for it in b_items[:pick_count]:
            if it["pair_id"] not in selected_pids:
                selected_real.append(it)
                selected_pids.add(it["pair_id"])

    # Fill remainder up to requested count
    remaining = [it for it in clean_items if it["pair_id"] not in selected_pids]
    rng.shuffle(remaining)
    needed = count - len(selected_real)
    selected_real.extend(remaining[:needed])

    assert len(selected_real) == count, f"Expected {count} real items, got {len(selected_real)}"
    selected_real.sort(key=lambda x: x.get("pair_id", ""))

    candidate_records = []
    item_num = 1

    for it in selected_real:
        txt = it.get("customer_text", "")
        cust_id = it.get("customer_tweet_id", f"tw_{item_num}")
        conv_id = it.get("conversation_id", f"conv_{cust_id}")
        pid = it.get("pair_id", "")

        if pid in known_pi_pids:
            sug_intent = "product_inquiry"
        else:
            sug_intent = classify_text_heuristically(txt)

        sug_esc, sug_trig, sug_reas = determine_escalation_heuristically(txt, sug_intent)

        # Estimate complexity
        if len(txt) > 150 or ("?" in txt and any(c in txt.lower() for c in ["and", "but", "however", "although"])):
            diff = "hard"
        elif len(txt) > 80:
            diff = "medium"
        else:
            diff = "easy"

        record = {
            "example_id": f"gold_{item_num:03d}",
            "candidate_id": f"cand_{item_num:03d}",
            "conversation_id": conv_id,
            "customer_tweet_id": str(cust_id),
            "customer_text": txt,
            "context": f"Customer inbound tweet to @AppleSupport (Recorded: {it.get('created_at', '2017')})",
            "source": "twcs",
            "split": "held_out",
            "support_reply": it.get("support_reply", ""),
            "machine_suggestion": {
                "intent": sug_intent,
                "expected_escalation": sug_esc,
                "escalation_trigger": sug_trig,
                "escalation_reason": sug_reas,
                "difficulty": diff
            }
        }
        candidate_records.append(record)
        item_num += 1

    # Append 20 targeted adversarial cases
    for adv in ADVERSARIAL_CASES:
        record = {
            "example_id": f"gold_{item_num:03d}",
            "candidate_id": f"cand_{item_num:03d}",
            "conversation_id": f"conv_adv_{item_num:03d}",
            "customer_tweet_id": f"adv_{item_num:03d}",
            "customer_text": adv["text"],
            "context": "Adversarial evaluation suite: safety hazard, legal, security, or explicit human request.",
            "source": "adversarial",
            "split": "held_out",
            "support_reply": adv["reply"],
            "machine_suggestion": {
                "intent": adv["intent"],
                "expected_escalation": adv["escalate"],
                "escalation_trigger": adv["trigger"],
                "escalation_reason": adv["reason"],
                "difficulty": adv["difficulty"]
            }
        }
        candidate_records.append(record)
        item_num += 1

    assert len(candidate_records) == 200, f"Expected 200 total candidates, got {len(candidate_records)}"

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for rec in candidate_records:
            f.write(json.dumps(rec) + "\n")

    with open(CANDIDATES_LEGACY_PATH, "w", encoding="utf-8") as f:
        for rec in candidate_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully generated {len(candidate_records)} candidates in {CANDIDATES_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified Candidate Sampling for Golden Set")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    parser.add_argument("--count", type=int, default=180, help="Number of real held-out TWCS candidates (default: 180)")
    args = parser.parse_args()
    sample_candidates(seed=args.seed, count=args.count)
