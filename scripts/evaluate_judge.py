"""
LLM Judge Evaluation & Human Validation Pipeline.

Evaluates the exact 45 production agent replies against authentic human ratings.
Calculates dimension-by-dimension and overall comparison metrics:
- Mean Absolute Error (MAE)
- Spearman rank correlation
- Pearson correlation
- Exact agreement rate
- Within +-1 point agreement rate

Fails closed if human_ratings.json is missing or incomplete.

Usage:
    python scripts/evaluate_judge.py
    python scripts/evaluate_judge.py --offline
    python scripts/evaluate_judge.py --live
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    HUMAN_RATINGS_JSON_PATH,
    GENERATED_REPLIES_PATH,
    LLM_JUDGE_RATINGS_PATH,
    JUDGE_VALIDATION_RESULTS_PATH,
    JUDGE_VALIDATION_DATA_PATH,
    JUDGE_MODEL_NAME,
    AGENT_MODEL_NAME,
    TARGET_BRAND
)
from src.llm_judge import LLMJudge

RATING_DIMENSIONS = [
    "groundedness",
    "helpfulness",
    "relevance",
    "brand_alignment",
    "safety",
    "overall"
]

def print_fail_closed_error():
    print(
        "\nHuman judge validation unavailable:\n"
        "human ratings have not been supplied.\n\n"
        "To enable judge validation:\n"
        "1. Run python scripts/export_human_review.py\n"
        "2. Manually populate ratings in data/judge/human_review.csv\n"
        "3. Run python scripts/import_human_ratings.py\n",
        file=sys.stderr
    )

def compute_dimension_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Computes honest ordinal metrics between human true scores and judge predicted scores."""
    a1 = np.array(y_true, dtype=float)
    a2 = np.array(y_pred, dtype=float)

    # 1. MAE
    mae = float(np.mean(np.abs(a1 - a2)))

    # 2. Exact agreement
    exact = float(np.mean(np.round(a1).astype(int) == np.round(a2).astype(int)))

    # 3. Within +- 1 point
    within_1 = float(np.mean(np.abs(a1 - a2) <= 1.0))

    # 4. Pearson correlation
    if np.std(a1) > 1e-6 and np.std(a2) > 1e-6:
        pearson_r = float(np.corrcoef(a1, a2)[0, 1])
    else:
        pearson_r = 1.0 if np.allclose(a1, a2) else 0.0

    # 5. Spearman rank correlation
    if np.std(a1) > 1e-6 and np.std(a2) > 1e-6:
        try:
            from scipy.stats import spearmanr
            spearman_rho, _ = spearmanr(a1, a2)
            spearman_rho = float(spearman_rho)
            if np.isnan(spearman_rho):
                spearman_rho = 0.0
        except Exception:
            spearman_rho = pearson_r
    else:
        spearman_rho = 1.0 if np.allclose(a1, a2) else 0.0

    return {
        "mae": round(mae, 2),
        "spearman": round(spearman_rho, 3),
        "pearson": round(pearson_r, 3),
        "exact_agreement": round(exact, 3),
        "within_one": round(within_1, 3)
    }

def format_evidence_context(evidence_list: List[Dict[str, Any]]) -> str:
    if not evidence_list:
        return "No historical evidence retrieved."
    lines = []
    for ev in evidence_list:
        pid = ev.get("pair_id", "N/A")
        q = (ev.get("query") or "").replace("\n", " ")
        r = (ev.get("response") or "").replace("\n", " ")
        lines.append(f"[{pid}] Cust: {q} | Agent: {r}")
    return "\n".join(lines)

def evaluate_judge(offline: bool = False):
    # 1. Check fail-closed condition: human ratings must exist
    if not HUMAN_RATINGS_JSON_PATH.exists():
        print_fail_closed_error()
        sys.exit(1)

    with open(HUMAN_RATINGS_JSON_PATH, "r", encoding="utf-8") as f:
        human_ratings = json.load(f)

    if not human_ratings or len(human_ratings) != 45:
        print_fail_closed_error()
        print(f"Error: human_ratings.json contains {len(human_ratings)} records, expected exactly 45.", file=sys.stderr)
        sys.exit(1)

    # Verify that all ratings have rater_type = "human"
    for r in human_ratings:
        if r.get("rater_type") != "human":
            print("Error: human_ratings.json contains non-human rater_type!", file=sys.stderr)
            sys.exit(1)

    human_by_id = {r["example_id"]: r for r in human_ratings}

    # 2. Load generated replies
    if not GENERATED_REPLIES_PATH.exists():
        raise FileNotFoundError(f"Generated replies not found at {GENERATED_REPLIES_PATH}. Run scripts/generate_judge_samples.py.")

    generated_records = []
    with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generated_records.append(json.loads(line))

    if len(generated_records) != 45:
        raise ValueError(f"Generated replies count is {len(generated_records)}, expected 45.")

    # Match IDs
    gen_by_id = {r["example_id"]: r for r in generated_records}
    if set(gen_by_id.keys()) != set(human_by_id.keys()):
        raise ValueError("Mismatch between generated replies IDs and human ratings IDs.")

    # 3. Evaluate each reply with LLM Judge
    print(f"Evaluating 45 agent replies with LLM Judge (offline={offline}, model={JUDGE_MODEL_NAME})...")
    judge = LLMJudge(model_name=JUDGE_MODEL_NAME)
    judge_records = []

    for item in generated_records:
        eid = item["example_id"]
        customer_text = item.get("customer_text", "")
        agent_reply = item.get("agent_reply") or "I am escalating this inquiry to a senior support advisor."
        context = item.get("context", f"Customer tweet to @{TARGET_BRAND}")
        evidence_context = format_evidence_context(item.get("retrieved_evidence", []))

        judge_res = judge.evaluate_reply(
            customer_text=customer_text,
            agent_reply=agent_reply,
            intent=item.get("predicted_intent", "other"),
            escalation_decision="escalate" if item.get("predicted_escalation") else "auto_handle",
            escalation_reason=item.get("escalation_reason", ""),
            context=context,
            historical_evidence=evidence_context,
            offline=offline
        )

        dim_scores = judge_res.get("dimension_scores", {})
        overall_score = judge_res.get("overall_score", 4.0)

        judge_records.append({
            "example_id": eid,
            "agent_reply": agent_reply,
            "groundedness": dim_scores.get("groundedness", 4),
            "helpfulness": dim_scores.get("helpfulness", 4),
            "relevance": dim_scores.get("relevance", 4),
            "brand_alignment": dim_scores.get("brand_alignment", 4),
            "safety": dim_scores.get("safety", 5),
            "overall": overall_score,
            "rationale": judge_res.get("rationale", "")
        })

    # Save judge scores
    LLM_JUDGE_RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_JUDGE_RATINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(judge_records, f, indent=2)
    print(f"Saved LLM judge ratings to {LLM_JUDGE_RATINGS_PATH}")

    # 4. Compare Human vs Judge
    dimension_results = {}
    judge_by_id = {j["example_id"]: j for j in judge_records}

    # Ordered list of IDs
    eids = sorted(human_by_id.keys())

    for dim in RATING_DIMENSIONS:
        h_vals = [human_by_id[eid][dim] for eid in eids]
        j_vals = [judge_by_id[eid][dim] for eid in eids]
        dimension_results[dim] = compute_dimension_metrics(h_vals, j_vals)

    # Overall aggregate metrics
    h_overall = [human_by_id[eid]["overall"] for eid in eids]
    j_overall = [judge_by_id[eid]["overall"] for eid in eids]
    overall_metrics = compute_dimension_metrics(h_overall, j_overall)

    validation_output = {
        "status": "completed",
        "sample_size": len(eids),
        "annotator_type": "human_single_annotator",
        "judge_model": JUDGE_MODEL_NAME,
        "generator_model": AGENT_MODEL_NAME,
        "independence_limitation": (
            "Judge and generator models belong to the same Gemini model family. "
            "The judge is therefore calibrated rather than fully independent."
        ),
        "dimensions": dimension_results,
        "overall": overall_metrics
    }

    # Save to both results/ and data/judge/
    JUDGE_VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_VALIDATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_output, f, indent=2)

    JUDGE_VALIDATION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_VALIDATION_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_output, f, indent=2)

    print(f"Exported judge validation metrics to {JUDGE_VALIDATION_RESULTS_PATH}")

    # Print summary table
    print("\n" + "=" * 65)
    print("LLM JUDGE VALIDATION RESULTS (Human vs Judge, N=45)")
    print("=" * 65)
    print(f"{'Dimension':<18} | {'MAE':<6} | {'Exact':<6} | {'Within+-1':<9} | {'Spearman':<8} | {'Pearson':<7}")
    print("-" * 65)
    for dim, met in dimension_results.items():
        print(f"{dim:<18} | {met['mae']:<6.2f} | {met['exact_agreement']*100:<5.1f}% | {met['within_one']*100:<8.1f}% | {met['spearman']:<8.3f} | {met['pearson']:<7.3f}")
    print("-" * 65)
    print(f"{'OVERALL':<18} | {overall_metrics['mae']:<6.2f} | {overall_metrics['exact_agreement']*100:<5.1f}% | {overall_metrics['within_one']*100:<8.1f}% | {overall_metrics['spearman']:<8.3f} | {overall_metrics['pearson']:<7.3f}")
    print("=" * 65 + "\n")

    return validation_output

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM Judge against human ratings.")
    parser.add_argument("--offline", action="store_true", help="Run deterministic offline heuristic judge.")
    parser.add_argument("--live", action="store_true", help="Run live Gemini LLM judge.")
    args = parser.parse_args()

    offline_mode = args.offline or not args.live
    evaluate_judge(offline=offline_mode)

if __name__ == "__main__":
    main()
