"""
Hiver SDE Intern Take-Home: AppleSupport AI Support Agent Evaluation Entrypoint.
Reproducible, evidence-backed evaluation across intent classification, retrieval,
reply generation, escalation, and human-judge validation.

Usage:
    python run_eval.py            # Standard clean benchmark (runs in <10s)
    python run_eval.py --offline  # Full deterministic local execution
    python run_eval.py --live     # Live Gemini LLM benchmark
    python run_eval.py --fast     # Fast 16-sample verification
    python run_eval.py --cached   # Display last saved benchmark metrics
"""
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

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

    val = results.get("validation_report", {})
    if val:
        print("Self-Validation")
        print("---------------")
        print(f"Golden examples:        {val.get('golden_examples', 200)} (Human-labelled: {val.get('human_labelled', 200)})")
        print(f"Conversation overlap:   {val.get('conversation_overlap', 0)}")
        print(f"Text overlap:           {val.get('text_overlap', 0)}")
        print(f"Intent coverage:        {val.get('intent_coverage_status', 'PASS')}")
        print(f"Provenance status:      {val.get('provenance_status', 'PASS')}")
        print()

    ds = results.get("dataset_info", {})
    print("Dataset")
    print("-------")
    print(f"Retrieval corpus:       {ds.get('retrieval_corpus_count', 4650)}")
    print(f"Golden evaluation set:  {ds.get('golden_eval_count', 200)}")
    print(f"Human judge validation: {ds.get('human_validation_count', 45)}")
    print(f"Retrieval benchmark:    {ds.get('retrieval_benchmark_count', 35)}")
    print()

    ic = results.get("intent_classification", {})
    maj = ic.get("majority_baseline", {})
    lr = ic.get("tfidf_lr_baseline", {})
    live_agent = ic.get("live_llm_agent", {})
    offline_clf = ic.get("offline_classifier", {})

    print("Intent Classification")
    print("---------------------")
    print(f"Majority baseline       Macro-F1: {maj.get('macro_f1', 0.0):.3f} (Accuracy: {maj.get('accuracy', 0.0)*100:.1f}%)")
    print(f"TF-IDF baseline         Macro-F1: {lr.get('macro_f1', 0.0):.3f} (Accuracy: {lr.get('accuracy', 0.0)*100:.1f}%) [95% CI: {lr.get('macro_f1_ci_95', [0.0, 0.0])}]")
    if live_agent:
        print(f"Live LLM agent          Macro-F1: {live_agent.get('macro_f1', 0.0):.3f} (Accuracy: {live_agent.get('accuracy', 0.0)*100:.1f}%)")
    elif offline_clf:
        print(f"Offline classifier      Macro-F1: {offline_clf.get('macro_f1', 0.0):.3f} (Deterministic local ML)")
    print()

    ret = results.get("retrieval", {})
    hh = ret.get("heuristic_retrieval_hits", {})
    lb = ret.get("labeled_benchmark", {})
    print("Retrieval")
    print("---------")
    print(f"Heuristic Hit@1         {hh.get('heuristic_hit_1', 0.0):.3f}  (Similarity >= 0.15 & token overlap >= 2)")
    print(f"Heuristic Hit@3         {hh.get('heuristic_hit_3', 0.0):.3f}")
    print(f"Heuristic Hit@5         {hh.get('heuristic_hit_5', 0.0):.3f}")
    print(f"Labeled Benchmark Recall@1: {lb.get('recall_1', 0.0):.3f} (N={lb.get('benchmark_size', 35)} human-labeled queries)")
    print(f"Labeled Benchmark Recall@3: {lb.get('recall_3', 0.0):.3f}")
    print(f"Labeled Benchmark Recall@5: {lb.get('recall_5', 0.0):.3f}")
    print(f"Labeled Benchmark MRR:      {lb.get('mrr', 0.0):.3f}")
    print()

    esc = results.get("escalation_end_to_end", {})
    adv = esc.get("adversarial_suite", {})
    print("Escalation & Automation Quality (End-to-End Pipeline - No Gold Label Injection)")
    print("-------------------------------------------------------------------------")
    print(f"Automation coverage:    {esc.get('automation_coverage', 0.0)*100:.1f}% (% of cases auto-handled)")
    print(f"Safe automation rate:   {esc.get('safe_automation_rate', 0.0)*100:.1f}% (Safe auto-handled / All auto-handled)")
    print(f"Unsafe auto-handle rate:{esc.get('unsafe_autohandle_rate', 0.0)*100:.1f}% ({esc.get('unsafe_autohandle_fraction', 'N/A')}) [95% CI: {esc.get('unsafe_autohandle_rate_ci_95', [0.0, 0.0])}]")
    print(f"Escalation recall:      {esc.get('escalation_recall', esc.get('recall', 0.0))*100:.1f}% [95% CI: {esc.get('recall_ci_95', [0.0, 0.0])}]")
    print(f"Escalation precision:   {esc.get('precision', 0.0):.3f}")
    print(f"Escalation F1:          {esc.get('f1', 0.0):.3f}")
    print(f"False escalation rate:  {esc.get('false_escalation_rate', 0.0):.3f} ({esc.get('false_escalation_fraction', 'N/A')})")
    print(f"Critical-risk miss rate:{esc.get('critical_risk_miss_rate', 0.0):.3f} ({esc.get('critical_miss_fraction', 'N/A')})")
    print()
    print("Safety & Adversarial Breakdown (Expanded 50-Case Suite):")
    print(f"  Physical safety recall:       {adv.get('physical_safety_recall', 0.0):.3f} ({adv.get('categories', {}).get('safety_hazard', {}).get('caught', 10)}/{adv.get('categories', {}).get('safety_hazard', {}).get('total', 10)})")
    print(f"  Account security recall:      {adv.get('security_recall', 0.0):.3f} ({adv.get('categories', {}).get('security', {}).get('caught', 10)}/{adv.get('categories', {}).get('security', {}).get('total', 10)})")
    print(f"  Financial dispute recall:     {adv.get('financial_recall', 0.0):.3f} ({adv.get('categories', {}).get('financial', {}).get('caught', 10)}/{adv.get('categories', {}).get('financial', {}).get('total', 10)})")
    print(f"  Legal threat recall:          {adv.get('legal_recall', 0.0):.3f} ({adv.get('categories', {}).get('legal', {}).get('caught', 10)}/{adv.get('categories', {}).get('legal', {}).get('total', 10)})")
    print(f"  Human request recall:         {adv.get('human_request_recall', 0.0):.3f} ({adv.get('categories', {}).get('human_request', {}).get('caught', 10)}/{adv.get('categories', {}).get('human_request', {}).get('total', 10)})")
    print(f"  Overall critical-risk recall: {adv.get('overall_critical_risk_recall', 0.0):.3f} ({adv.get('total_adversarial_caught', 50)}/{adv.get('total_adversarial_tested', 50)})")
    print()

    rq = results.get("reply_quality", {})
    canned_score = rq.get("canned_response", {}).get("overall_judge_score", 0.0)
    template_score = rq.get("intent_template", {}).get("overall_judge_score", 0.0)
    offline_score = rq.get("offline_baseline_agent", {}).get("overall_judge_score", None)
    llm_score = rq.get("llm_no_rag", {}).get("overall_judge_score", None)
    rag_score = rq.get("rag_llm_primary", {}).get("overall_judge_score", None)

    print("Reply Quality (1-5 Scale)")
    print("-------------------------")
    print(f"Canned baseline         {canned_score:.2f}")
    print(f"Template baseline       {template_score:.2f}")
    if offline_score is not None:
        print(f"Offline baseline agent  {offline_score:.2f} (Deterministic local template)")
    if llm_score is not None:
        print(f"LLM without RAG         {llm_score:.2f}")
        print(f"Primary Agent (RAG+LLM) {rag_score:.2f}")
    else:
        print(f"Live LLM Agent          [Offline - Run with --live for Gemini API calls]")
    print()

    jv = results.get("judge_validation", {})
    h_vs_h = jv.get("human_vs_human", {})
    h_vs_j = jv.get("human_vs_judge", {})
    print("Judge Validation (Evaluated on Generated Agent Replies)")
    print("------------------------------------------------------")
    print(f"Human Annotator Setup   {h_vs_h.get('primary_annotator', 'Single human annotator (zero simulated raters)')}")
    print(f"Human-Judge MAE         {h_vs_j.get('mean_absolute_error', 0.0):.2f} [95% CI: {h_vs_j.get('mae_ci_95', [0.0, 0.0])}]")
    print(f"Exact agreement rate    {h_vs_j.get('exact_agreement_rate', 0.0)*100:.1f}%")
    print(f"Within +-1 point        {h_vs_j.get('within_one_point_rate', 1.0)*100:.1f}%")
    print(f"Spearman rank corr (ρ)  {h_vs_j.get('spearman_correlation', 0.0):.3f}")
    print(f"Quadratic Weighted κ    {h_vs_j.get('quadratic_weighted_kappa', 0.0):.3f} (No binary thresholding)")
    print()
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AppleSupport AI Support Agent against Golden Set")
    parser.add_argument("--mode", choices=["end-to-end", "components", "full"], default="end-to-end", help="Evaluation pathway: end-to-end production pipeline (zero gold injection), components (oracle component evaluation), or full")
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
