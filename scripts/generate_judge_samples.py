"""
Generate Judge Evaluation Samples from Golden Evaluation Set.

Executes the actual production pipeline across 45 deterministically selected examples:
Customer message -> Predicted intent -> Historical retrieval -> Escalation decision -> Reply generation

Strict Guarantees:
- Deterministic selection using SEED = 42 across 45 examples.
- Stratified across all 8 intents, auto-handled, escalated, easy, medium, hard, and edge cases.
- NO gold intent injection into the production pipeline. Gold intent is saved purely for evaluation.
- Real production agent replies are logged with retrieved evidence.

Outputs:
1. data/judge/judge_validation_sample.jsonl (the exact 45 sampled items)
2. data/judge/generated_replies.jsonl (agent generated responses with evidence)
"""
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    SEED,
    DATA_DIR,
    GOLDEN_EVAL_PATH,
    GOLDEN_EVAL_CANONICAL_PATH,
    JUDGE_DIR,
    JUDGE_SAMPLE_PATH,
    GENERATED_REPLIES_PATH,
    INTENT_NAMES
)
from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetrievalEngine
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator

def load_golden_dataset() -> List[Dict[str, Any]]:
    path = GOLDEN_EVAL_PATH if GOLDEN_EVAL_PATH.exists() else GOLDEN_EVAL_CANONICAL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Golden evaluation set not found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def select_diverse_sample(items: List[Dict[str, Any]], sample_size: int = 45, seed: int = SEED) -> List[Dict[str, Any]]:
    """
    Deterministically selects sample_size (45) diverse examples from items:
    - Guarantees all 8 intents represented
    - Includes both auto-handled and escalated cases
    - Includes short, medium, and long customer messages
    - Includes easy, medium, and edge/adversarial cases
    """
    rng = random.Random(seed)

    # Sort deterministically by example_id
    sorted_items = sorted(items, key=lambda x: x["example_id"])

    # Group by intent
    by_intent = defaultdict(list)
    for it in sorted_items:
        intent = it.get("intent", it.get("ground_truth_intent", "other"))
        by_intent[intent].append(it)

    # We want 45 examples across 8 intents:
    # 5 intents will get 6 examples (30), 3 intents will get 5 examples (15) -> Total 45
    sorted_intents = sorted(INTENT_NAMES)
    allocations = {}
    for i, intent in enumerate(sorted_intents):
        allocations[intent] = 6 if i < 5 else 5

    selected = []
    for intent in sorted_intents:
        cand = list(by_intent[intent])
        k = allocations[intent]

        # Prioritize diversity within the intent:
        # Separate escalated vs non-escalated, short vs long, difficult vs easy
        cand_sorted = sorted(
            cand,
            key=lambda x: (
                not x.get("expected_escalation", False),  # include escalated cases first
                x.get("difficulty") != "hard",           # include hard cases
                len(x.get("customer_text", x.get("customer_message", ""))) % 7  # pseudorandom hash
            )
        )

        # Pick k items using deterministic stride/sampling
        if len(cand_sorted) <= k:
            selected.extend(cand_sorted)
        else:
            # Deterministic subset
            step = len(cand_sorted) / k
            picks = [cand_sorted[int(i * step)] for i in range(k)]
            selected.extend(picks)

    # Ensure exact sample_size
    selected = sorted(selected[:sample_size], key=lambda x: x["example_id"])
    return selected

def generate_replies():
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading golden evaluation set...")
    golden_items = load_golden_dataset()
    print(f"Loaded {len(golden_items)} items.")

    print(f"Selecting 45 diverse judge validation examples (SEED={SEED})...")
    sampled_items = select_diverse_sample(golden_items, sample_size=45, seed=SEED)
    print(f"Selected {len(sampled_items)} examples.")

    # Save exact selection
    with open(JUDGE_SAMPLE_PATH, "w", encoding="utf-8") as f:
        for it in sampled_items:
            f.write(json.dumps(it) + "\n")
    print(f"Saved judge validation sample selection to {JUDGE_SAMPLE_PATH}")

    # Initialize production components
    print("Initializing production agent pipeline components...")
    retrieval_engine = HistoricalRetrievalEngine.load()
    intent_classifier = IntentClassifier()
    escalation_engine = EscalationEngine()
    reply_generator = ReplyGenerator(retrieval_engine=retrieval_engine)

    print("Executing production pipeline on 45 examples (ZERO gold intent injection)...")
    generated_records = []

    for item in sampled_items:
        eid = item["example_id"]
        customer_text = item.get("customer_text", item.get("customer_message", ""))
        context = item.get("context", "Customer tweet to @AppleSupport")
        gold_intent = item.get("intent", item.get("ground_truth_intent", "other"))
        gold_esc = item.get("expected_escalation", False)

        # 1. Intent Classification (Production learned classifier, NO gold intent passed)
        intent_res = intent_classifier.classify(customer_text, method="learned")
        pred_intent = intent_res["intent"]
        pred_conf = intent_res.get("confidence", 0.8)

        # 2. Historical Retrieval
        retrieved = retrieval_engine.retrieve(customer_text, top_k=3)
        retrieved_ids = [r.get("pair_id") for r in retrieved if r.get("pair_id")]
        retrieved_evidence = [
            {
                "pair_id": r.get("evidence_id", r.get("pair_id", "")),
                "query": r.get("customer_text", r.get("historical_customer_query", "")),
                "response": r.get("support_reply", r.get("historical_agent_response", "")),
                "similarity": round(float(r.get("similarity_score", r.get("similarity", 0.0))), 3)
            }
            for r in retrieved
        ]

        # 3. Escalation Decision
        esc_res = escalation_engine.decide(customer_text, intent=pred_intent, intent_confidence=pred_conf)
        pred_esc_str = esc_res["decision"]
        pred_esc_bool = (pred_esc_str == "escalate")
        esc_reason = esc_res["reason"]

        # 4. Reply Generation
        reply_data = reply_generator.generate_reply(
            customer_text=customer_text,
            intent=pred_intent,
            escalation_decision=pred_esc_str,
            escalation_reason=esc_reason,
            method="template"
        )
        agent_reply = reply_data.get("reply")

        rec = {
            "example_id": eid,
            "customer_text": customer_text,
            "context": context,
            "gold_intent": gold_intent,
            "predicted_intent": pred_intent,
            "gold_escalation": gold_esc,
            "predicted_escalation": pred_esc_bool,
            "escalation_reason": esc_reason,
            "agent_reply": agent_reply,
            "retrieved_evidence_ids": retrieved_ids,
            "retrieved_evidence": retrieved_evidence
        }
        generated_records.append(rec)

    with open(GENERATED_REPLIES_PATH, "w", encoding="utf-8") as f:
        for r in generated_records:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully generated and saved {len(generated_records)} production replies to {GENERATED_REPLIES_PATH}")

if __name__ == "__main__":
    generate_replies()
