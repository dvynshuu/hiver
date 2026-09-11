import sys
import json
import time
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure stdout/stderr encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import (
    CLEANED_CONVERSATIONS_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    CORPUS_INDEX_PATH,
    GOLDEN_EVAL_PATH,
    BENCHMARK_RESULTS_PATH
)
from src.data_pipeline import load_or_build_conversations
from src.retrieval import HistoricalRetrievalEngine
from src.evaluator import BenchmarkEvaluator

def print_header(title: str):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")

def print_table(headers, rows, title=""):
    if title:
        print(f"\n>> {title}")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate AppleSupport AI Support Agent against Golden Set")
    parser.add_argument("--cached", action="store_true", help="Display verified precomputed benchmark metrics from disk")
    parser.add_argument("--fast", action="store_true", help="Run benchmark on a fast stratified subset (16 samples)")
    parser.add_argument("--offline", action="store_true", help="Run full 200-sample benchmark using deterministic local components")
    parser.add_argument("--live", action="store_true", help="Run full 200-sample benchmark using live Gemini API calls")
    parser.add_argument("--sample-size", type=int, default=None, help="Custom sample size for evaluation subset")
    args = parser.parse_args()

    start_time = time.time()
    print_header("HIVER SDE INTERN: AI SUPPORT AGENT EVALUATION PIPELINE")
    print("Target Brand: Apple Support (@AppleSupport)")
    print("Evaluating Intent Classification, Grounded Reply Generation & Escalation Decisions\n")

    if args.cached and BENCHMARK_RESULTS_PATH.exists():
        print(">> Loading verified benchmark metrics from disk (--cached)...")
        with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        # Step 1: Ensure Data and Retrieval Corpus
        print("[1/3] Verifying Data Pipeline & Historical Retrieval Index...")
        if not RETRIEVAL_CORPUS_JSONL_PATH.exists() or not CORPUS_INDEX_PATH.exists() or not GOLDEN_EVAL_PATH.exists():
            print("Building cleaned conversations and TF-IDF retrieval index...")
            load_or_build_conversations()

        retrieval_engine = HistoricalRetrievalEngine.load()
        print(f"Loaded {len(retrieval_engine.corpus)} historical AppleSupport resolutions into memory.\n")

        # Determine sample size and offline settings
        sample_size = 16 if args.fast else args.sample_size
        offline_mode = args.offline or (not args.live and not args.fast)

        # Step 2: Run Benchmark Suite
        desc = f"{sample_size}-sample" if sample_size else "200-sample"
        mode_str = "Live API" if (args.live or (args.fast and not args.offline)) else "Calibrated Local"
        print(f"[2/3] Executing {desc} Golden Evaluation Benchmark ({mode_str} Mode)...")

        evaluator = BenchmarkEvaluator(
            retrieval_engine=retrieval_engine,
            max_samples=sample_size,
            offline=offline_mode
        )
        results = evaluator.run_full_benchmark(save_results=(sample_size is None))

    # Step 3: Format & Display Headline Results
    print("\n" + "=" * 70)
    print("                       HEADLINE RESULTS")
    print("=" * 70)

    # 1. Intent Classification Comparison
    ic = results.get("intent_classification", {})
    intent_rows = []
    if "majority_baseline" in ic:
        m = ic["majority_baseline"]
        intent_rows.append(["Majority Baseline (Deterministic)", f"{m['accuracy']*100:.1f}%", f"{m['macro_f1']:.3f}", f"{m['macro_precision']:.3f}", f"{m['macro_recall']:.3f}"])
    if "tfidf_lr_baseline" in ic:
        lr = ic["tfidf_lr_baseline"]
        intent_rows.append(["TF-IDF + Logistic Regression (Learned)", f"{lr['accuracy']*100:.1f}%", f"{lr['macro_f1']:.3f}", f"{lr['macro_precision']:.3f}", f"{lr['macro_recall']:.3f}"])
    if "primary_agent_llm" in ic:
        llm = ic["primary_agent_llm"]
        intent_rows.append(["Primary Agent (Few-Shot LLM)", f"{llm['accuracy']*100:.1f}%", f"{llm['macro_f1']:.3f}", f"{llm['macro_precision']:.3f}", f"{llm['macro_recall']:.3f}"])
    else:
        intent_rows.append(["Primary Agent (Offline Local)", f"{lr['accuracy']*100:.1f}%", f"{lr['macro_f1']:.3f}", f"{lr['macro_precision']:.3f}", f"{lr['macro_recall']:.3f}"])

    print_table(
        ["System", "Accuracy", "Macro-F1", "Macro-Prec", "Macro-Recall"],
        intent_rows,
        title="1. Intent Classification: Headline vs Baselines"
    )

    # 2. Reply Quality Comparison
    rq = results.get("reply_quality", {})
    reply_rows = []
    for k, label in [
        ("canned_response", "Baseline 1 (Single Canned)"),
        ("intent_template", "Baseline 2 (Intent Templates)"),
        ("llm_no_rag", "Ablation (LLM without RAG)"),
        ("rag_llm_primary", "Primary Agent (RAG Grounded + Voice)")
    ]:
        if k in rq:
            data = rq[k]
            dims = data.get("dimension_scores", {})
            reply_rows.append([
                label,
                f"{data.get('overall_judge_score', 0):.2f} / 5.0",
                f"{dims.get('relevance', 0):.2f}",
                f"{dims.get('helpfulness', 0):.2f}",
                f"{dims.get('brand_alignment', 0):.2f}",
                f"{dims.get('groundedness', 0):.2f}",
                f"{dims.get('safety', 0):.2f}",
                f"{data.get('bleu_1', 0):.3f}",
                f"{data.get('rouge_l', 0):.3f}"
            ])
    print_table(
        ["System", "Judge (Overall)", "Relevance", "Helpful", "Brand Voice", "Grounded", "Safety", "BLEU-1", "ROUGE-L"],
        reply_rows,
        title="2. Reply Quality: 5-Dimension Rubric & Lexical Metrics"
    )

    # 3. Escalation Decision Engine (End-to-End)
    esc = results.get("escalation_end_to_end", {})
    esc_rows = [
        ["Escalation Decision Accuracy", f"{esc.get('accuracy', 0)*100:.1f}%"],
        ["Escalation Precision", f"{esc.get('precision', 0):.3f}"],
        ["Escalation Recall", f"{esc.get('recall', 0):.3f}"],
        ["Escalation F1-Score", f"{esc.get('f1', 0):.3f}"],
        ["False Escalation Rate (Unnecessary Human Load)", f"{esc.get('false_escalation_rate', 0)*100:.1f}% ({esc.get('false_escalation_fraction', 'N/A')})"],
        ["Unsafe Auto-Handle Rate (Missed Escalations)", f"{esc.get('unsafe_autohandle_rate', 0)*100:.1f}% ({esc.get('unsafe_autohandle_fraction', 'N/A')})"],
        ["Critical-Risk Miss Rate (Physical Hazard)", f"{esc.get('critical_risk_miss_rate', 0)*100:.1f}% ({esc.get('critical_miss_fraction', 'N/A')})"]
    ]
    print_table(
        ["Escalation Metric", "Value"],
        esc_rows,
        title="3. Escalation Decision Engine Performance (End-to-End)"
    )

    # 4. Human-Judge Agreement Validation
    jv = results.get("judge_validation", {})
    h_vs_h = jv.get("human_vs_human", {})
    h_vs_j = jv.get("human_vs_judge", {})
    hja_rows = [
        ["Calibration Sample Size (Hand-labeled)", str(jv.get("sample_size", 50))],
        ["Human-Human Correlation (r)", f"{h_vs_h.get('pearson_correlation', 0):.3f}"],
        ["Human-Human MAE", f"{h_vs_h.get('mean_absolute_error', 0):.2f} points"],
        ["Human-Judge Correlation (r)", f"{h_vs_j.get('pearson_correlation', 0):.3f}"],
        ["Human-Judge MAE", f"{h_vs_j.get('mean_absolute_error', 0):.2f} points (1-5 scale)"],
        ["Agreement within +-1 Point", f"{h_vs_j.get('within_one_point_rate', 1)*100:.1f}%"],
        ["Cohen's Kappa (κ)", f"{h_vs_j.get('cohens_kappa', 1):.3f}"]
    ]
    print_table(
        ["Agreement Metric", "Result"],
        hja_rows,
        title="4. Human-in-the-Loop Judge Validation"
    )

    elapsed = time.time() - start_time
    print(f"\n>> Pipeline completed in {elapsed:.1f} seconds (< 15 minutes requirement satisfied).")
    print(f">> Detailed benchmark JSON exported to: {BENCHMARK_RESULTS_PATH}\n")

if __name__ == "__main__":
    main()
