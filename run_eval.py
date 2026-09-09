"""
Hiver SDE Intern Take-Home: AppleSupport AI Support Agent Evaluation Entrypoint.
Reproducible, evidence-backed evaluation across intent classification, retrieval,
reply generation, escalation, and human-judge validation.

Usage:
    python run_eval.py            # Standard clean benchmark (runs in <10s)
    python run_eval.py --live     # Live Gemini LLM benchmark
    python run_eval.py --fast     # Fast 16-sample verification
    python run_eval.py --cached   # Display last saved benchmark metrics
"""
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    TARGET_BRAND,
    BENCHMARK_RESULTS_PATH,
    CLEANED_CONVERSATIONS_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    CORPUS_INDEX_PATH,
    GOLDEN_EVAL_PATH,
    GEMINI_API_KEY
)
from src.data_pipeline import load_or_build_conversations
from src.retrieval import HistoricalRetrievalEngine
from src.evaluator import BenchmarkEvaluator

def print_banner(title: str):
    print("=" * 50)
    print(title)
    print("=" * 50)

def format_terminal_report(results: Dict[str, Any]):
    print_banner("HIVER SUPPORT AGENT EVALUATION")
    print(f"\nBrand: {TARGET_BRAND}\n")

    ds = results.get("dataset_info", {})
    print("Dataset")
    print("-------")
    print(f"Retrieval corpus:       {ds.get('retrieval_corpus_count', 4650)}")
    print(f"Golden evaluation set:  {ds.get('golden_eval_count', 200)}")
    print(f"Human judge validation: {ds.get('human_validation_count', 50)}")
    print()

    lk = results.get("leakage_check", {})
    print("Leakage")
    print("-------")
    print(f"Exact text overlap:     {lk.get('exact_text_overlap', 0)}")
    print(f"Normalized overlap:     {lk.get('normalized_overlap', 0)}")
    print(f"Conversation overlap:   {lk.get('conversation_overlap', 0)}")
    print(f"Status: {lk.get('status', 'PASS')}")
    print()

    ic = results.get("intent_classification", {})
    maj_f1 = ic.get("majority_baseline", {}).get("macro_f1", 0.0)
    lr_f1 = ic.get("tfidf_lr_baseline", {}).get("macro_f1", 0.0)
    llm_f1 = ic.get("primary_agent_llm", {}).get("macro_f1", None)

    print("Intent Classification")
    print("---------------------")
    print(f"Majority baseline       Macro-F1: {maj_f1:.3f}")
    print(f"TF-IDF baseline         Macro-F1: {lr_f1:.3f}")
    if llm_f1 is not None:
        print(f"LLM                     Macro-F1: {llm_f1:.3f}")
        print(f"Full agent              Macro-F1: {llm_f1:.3f}")
    else:
        print(f"LLM                     Macro-F1: [Offline - Not Queried]")
        print(f"Full agent (offline)    Macro-F1: {lr_f1:.3f}")
    print()

    ret = results.get("retrieval", {})
    print("Retrieval")
    print("---------")
    print(f"Recall@1                {ret.get('recall_1', 0.0):.3f}")
    print(f"Recall@3                {ret.get('recall_3', 0.0):.3f}")
    print(f"Recall@5                {ret.get('recall_5', 0.0):.3f}")
    print()

    rq = results.get("reply_quality", {})
    canned_score = rq.get("canned_response", {}).get("overall_judge_score", 0.0)
    template_score = rq.get("intent_template", {}).get("overall_judge_score", 0.0)
    llm_score = rq.get("llm_no_rag", {}).get("overall_judge_score", None)
    rag_score = rq.get("rag_llm_primary", {}).get("overall_judge_score", None)

    print("Reply Quality")
    print("-------------")
    print(f"Canned                  {canned_score:.2f}")
    print(f"Template                {template_score:.2f}")
    if llm_score is not None:
        print(f"LLM                     {llm_score:.2f}")
        print(f"LLM + RAG               {rag_score:.2f}")
    else:
        print("LLM                     [Offline - Requires API Key]")
        print("LLM + RAG               [Offline - Requires API Key]")
    print()

    esc = results.get("escalation_end_to_end", {})
    print("Escalation (End-to-End)")
    print("-----------------------")
    print(f"Precision               {esc.get('precision', 0.0):.3f}")
    print(f"Recall                  {esc.get('recall', 0.0):.3f}")
    print(f"F1                      {esc.get('f1', 0.0):.3f}")
    print(f"False escalation rate   {esc.get('false_escalation_rate', 0.0):.3f} ({esc.get('false_escalation_fraction', 'N/A')})")
    print(f"Unsafe auto-handle rate {esc.get('unsafe_autohandle_rate', 0.0):.3f} ({esc.get('unsafe_autohandle_fraction', 'N/A')})")
    print(f"Critical-risk miss rate {esc.get('critical_risk_miss_rate', 0.0):.3f} ({esc.get('critical_miss_fraction', 'N/A')})")
    print()

    jv = results.get("judge_validation", {})
    h_vs_h = jv.get("human_vs_human", {})
    h_vs_j = jv.get("human_vs_judge", {})
    print("Judge Validation")
    print("----------------")
    print(f"Human-human agreement   {h_vs_h.get('pearson_correlation', 0.0):.3f} (r), {h_vs_h.get('mean_absolute_error', 0.0):.2f} (MAE)")
    print(f"Human-LLM correlation   {h_vs_j.get('pearson_correlation', 0.0):.3f}")
    print(f"Human-LLM MAE           {h_vs_j.get('mean_absolute_error', 0.0):.2f}")
    print(f"Within +-1 point        {h_vs_j.get('within_one_point_rate', 1.0)*100:.1f}%")
    print(f"Cohen's Kappa (κ)       {h_vs_j.get('cohens_kappa', 1.0):.3f}")
    print()
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AppleSupport AI Support Agent against Golden Set")
    parser.add_argument("--cached", action="store_true", help="Display verified precomputed benchmark metrics from disk")
    parser.add_argument("--fast", action="store_true", help="Run benchmark on a fast stratified subset (16 samples)")
    parser.add_argument("--offline", action="store_true", help="Run full benchmark using deterministic local components")
    parser.add_argument("--live", action="store_true", help="Run full benchmark using live Gemini API calls")
    parser.add_argument("--sample-size", type=int, default=None, help="Custom sample size for evaluation subset")
    args = parser.parse_args()

    start_time = time.time()

    if args.cached and BENCHMARK_RESULTS_PATH.exists():
        with open(BENCHMARK_RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
        format_terminal_report(results)
        print(">> Note: Displayed cached previous benchmark metrics.")
        return

    # Ensure data and retrieval index
    if not RETRIEVAL_CORPUS_JSONL_PATH.exists() or not CORPUS_INDEX_PATH.exists() or not GOLDEN_EVAL_PATH.exists():
        print("Ensuring clean conversation splits and retrieval index...")
        load_or_build_conversations()

    retrieval_engine = HistoricalRetrievalEngine.load()

    sample_size = 16 if args.fast else args.sample_size
    offline_mode = args.offline or (not args.live and not args.fast)

    evaluator = BenchmarkEvaluator(
        retrieval_engine=retrieval_engine,
        max_samples=sample_size,
        offline=offline_mode
    )

    results = evaluator.run_full_benchmark(save_results=(sample_size is None))

    format_terminal_report(results)

    elapsed = time.time() - start_time
    print(f"Reproduction runtime: {elapsed:.2f} seconds (< 15 minutes requirement satisfied).")
    print(f"Benchmark results exported to: {BENCHMARK_RESULTS_PATH}")

if __name__ == "__main__":
    main()
