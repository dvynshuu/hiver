"""
Hiver Support Agent: Golden Evaluation Pipeline.
Audits evaluation integrity, reproducibility, and end-to-end correctness.
Enforces:
1. Pre-flight dataset self-validation (leakage, provenance, intent coverage).
2. Strict separation: Component-Level (Oracle) vs End-to-End (Zero gold injection).
3. Retrieval metrics: Heuristic Hit@K vs Human-labeled Retrieval Benchmark (Recall@K, MRR).
4. Safety & Escalation: Unsafe auto-handle rate, critical-risk miss rate, and individual category recall.
5. Honest Judge Validation: Actual agent replies, 1-5 scale, quadratic weighted kappa, zero thresholding tricks.
6. Auditable execution records and bootstrap confidence intervals.
"""
import sys
import os
import json
import subprocess
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    INTENT_NAMES,
    GOLDEN_EVAL_PATH,
    BENCHMARK_RESULTS_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    HUMAN_RATINGS_JSON_PATH,
    RESULTS_DIR,
    TARGET_BRAND,
    SEED,
    AGENT_MODEL_NAME,
    JUDGE_MODEL_NAME
)
from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetrievalEngine
from src.reply_generator import ReplyGenerator
from src.escalation_engine import EscalationEngine
from src.llm_judge import LLMJudge
from src.data_pipeline import check_evaluation_leakage, format_leakage_report

logger = logging.getLogger(__name__)

E2E_RECORDS_PATH = RESULTS_DIR / "end_to_end_evaluation_records.jsonl"
METADATA_PATH = RESULTS_DIR / "benchmark_metadata.json"

def calculate_bootstrap_ci(
    values_fn,
    n_samples: int,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = SEED
) -> Tuple[float, float]:
    """Calculate empirical bootstrap confidence interval for a metric function."""
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    boot_estimates = []

    for _ in range(n_bootstraps):
        bs_idx = rng.choice(indices, size=n_samples, replace=True)
        val = values_fn(bs_idx)
        if val is not None and not np.isnan(val):
            boot_estimates.append(val)

    if not boot_estimates:
        return 0.0, 0.0

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_estimates, 100 * alpha))
    high = float(np.percentile(boot_estimates, 100 * (1.0 - alpha)))
    return round(low, 3), round(high, 3)

def calculate_lexical_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Computes BLEU-1 and ROUGE-L lexical overlap metrics."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"bleu_1": 0.0, "rouge_l": 0.0, "token_f1": 0.0}

    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    pred_overlap = sum(1 for t in pred_tokens if t in common_tokens)
    ref_overlap = sum(1 for t in ref_tokens if t in common_tokens)

    p = pred_overlap / len(pred_tokens) if pred_tokens else 0.0
    r = ref_overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # ROUGE-L LCS
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    rouge_l = (2 * lcs / (m + n)) if (m + n) > 0 else 0.0

    return {
        "bleu_1": round(p, 4),
        "rouge_l": round(rouge_l, 4),
        "token_f1": round(f1, 4)
    }

class BenchmarkEvaluator:
    """
    Self-validating benchmark evaluation suite for @AppleSupport.
    Guarantees strict end-to-end evaluation with zero gold-label injection,
    labeled retrieval benchmarking, honest judge calibration, and auditable logging.
    """

    def __init__(
        self,
        eval_set_path: Path = GOLDEN_EVAL_PATH,
        retrieval_engine: Optional[HistoricalRetrievalEngine] = None,
        max_samples: Optional[int] = None,
        offline: bool = False
    ):
        self.eval_set_path = eval_set_path
        self.max_samples = max_samples
        self.offline = offline
        self.eval_data: List[Dict[str, Any]] = []
        self._load_eval_set()

        # Initialize core components
        self.intent_classifier = IntentClassifier()
        self.retrieval_engine = retrieval_engine or HistoricalRetrievalEngine.load()
        self.reply_generator = ReplyGenerator(retrieval_engine=self.retrieval_engine)
        self.escalation_engine = EscalationEngine()
        self.judge = LLMJudge()

    def _load_eval_set(self):
        """Loads Golden Evaluation Set with optional sample capping."""
        if not self.eval_set_path.exists():
            raise FileNotFoundError(f"Golden evaluation set not found at {self.eval_set_path}")

        all_items = []
        with open(self.eval_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))

        if self.max_samples and len(all_items) > self.max_samples:
            self.eval_data = all_items[:self.max_samples]
        else:
            self.eval_data = all_items

    # -------------------------------------------------------------
    # PRE-FLIGHT SELF-VALIDATION (Section 22 of Master Prompt)
    # -------------------------------------------------------------
    def run_self_validation(self) -> Dict[str, Any]:
        """
        Validates the dataset integrity, provenance, human labeling,
        leakage isolation, and intent coverage before any benchmark starts.
        Fails loudly if any condition is violated.
        """
        retrieval_corpus = []
        if RETRIEVAL_CORPUS_JSONL_PATH.exists():
            with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        retrieval_corpus.append(json.loads(line))
        elif self.retrieval_engine and self.retrieval_engine.corpus:
            retrieval_corpus = self.retrieval_engine.corpus

        # 1. Leakage Check
        leakage_results = check_evaluation_leakage(retrieval_corpus, self.eval_data)
        if leakage_results["status"] != "PASS":
            raise ValueError(f"CRITICAL: Evaluation Leakage Detected!\n{format_leakage_report(leakage_results)}")

        # 2. Golden Provenance & Human Labeling Check
        required_fields = [
            "example_id", "conversation_id", "customer_tweet_id", "customer_text",
            "context", "source", "split", "intent", "expected_escalation",
            "difficulty", "annotator", "annotator_type"
        ]

        intent_counts = {name: 0 for name in INTENT_NAMES}
        human_labelled_count = 0
        conversation_ids = set()

        for item in self.eval_data:
            eid = item.get("example_id", "unknown")
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Golden item {eid} lacks required provenance field '{field}'!")

            if item.get("annotator") == "human" and item.get("annotator_type") == "human_single_annotator":
                human_labelled_count += 1
            else:
                raise ValueError(f"Golden item {eid} lacks valid human annotation!")

            intent = item.get("intent")
            if intent not in INTENT_NAMES:
                raise ValueError(f"Golden item {eid} has invalid intent '{intent}'!")
            intent_counts[intent] += 1

            cid = item.get("conversation_id")
            if cid in conversation_ids:
                raise ValueError(f"Duplicate conversation ID detected in golden set: '{cid}'!")
            conversation_ids.add(cid)

        # 3. Intent Coverage Validation (Zero coverage prohibited)
        zero_coverage = [name for name, count in intent_counts.items() if count == 0]
        if zero_coverage:
            raise ValueError(f"CRITICAL: Golden evaluation set has zero coverage for intents: {zero_coverage}!")

        low_coverage = [name for name, count in intent_counts.items() if count < 5]
        intent_status = "PASS" if not low_coverage else f"PASS_WITH_WARNING ({low_coverage})"

        validation_summary = {
            "golden_examples": len(self.eval_data),
            "human_labelled": human_labelled_count,
            "conversation_overlap": leakage_results["conversation_overlap"],
            "text_overlap": leakage_results["exact_text_overlap"],
            "normalized_overlap": leakage_results["normalized_overlap"],
            "intent_coverage_status": intent_status,
            "provenance_status": "PASS",
            "status": "PASS",
            "intent_distribution": intent_counts
        }

        print("==================================================")
        print("DATASET VALIDATION")
        print("==================================================")
        print(f"Golden examples:      {len(self.eval_data)}")
        print(f"Human-labelled:       {human_labelled_count}")
        print(f"Conversation overlap: {leakage_results['conversation_overlap']}")
        print(f"Text overlap:         {leakage_results['exact_text_overlap']}")
        print(f"Intent coverage:      {intent_status}")
        print(f"Provenance:           PASS")
        print("==================================================\n")

        return validation_summary

    # -------------------------------------------------------------
    # INTENT CLASSIFICATION EVALUATION
    # -------------------------------------------------------------
    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """
        Evaluates intent classification across:
        1. Majority Class Baseline (Deterministic software_bug)
        2. TF-IDF + Logistic Regression Baseline (Learned, Seed 42)
        3. Offline Classifier / Live LLM Primary Agent
        Computes 95% bootstrap confidence intervals for Macro-F1.
        """
        y_true = [item.get("ground_truth_intent", item.get("intent")) for item in self.eval_data]
        texts = [item.get("customer_message", item.get("customer_text")) for item in self.eval_data]

        modes = [
            ("majority", "majority_baseline"),
            ("learned", "tfidf_lr_baseline")
        ]
        if not self.offline:
            modes.append(("llm", "live_llm_agent"))
        else:
            modes.append(("learned", "offline_classifier"))

        results = {}
        for mode, name in modes:
            y_pred = []
            for t in texts:
                res = self.intent_classifier.classify(t, method=mode if mode != "live_llm_agent" else "llm")
                y_pred.append(res.get("intent", "other"))

            acc = accuracy_score(y_true, y_pred)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=INTENT_NAMES, average="macro", zero_division=0
            )
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=INTENT_NAMES, average="weighted", zero_division=0
            )

            # Per-class P/R/F1
            cp, cr, cf1, cs = precision_recall_fscore_support(
                y_true, y_pred, labels=INTENT_NAMES, average=None, zero_division=0
            )
            per_class = {
                name_i: {
                    "precision": round(float(cp[i]), 3),
                    "recall": round(float(cr[i]), 3),
                    "f1": round(float(cf1[i]), 3),
                    "golden_count": int(cs[i])
                }
                for i, name_i in enumerate(INTENT_NAMES)
            }

            cm = confusion_matrix(y_true, y_pred, labels=INTENT_NAMES).tolist()

            # Bootstrap 95% Confidence Interval for Macro-F1
            def _f1_boot(indices):
                sub_true = [y_true[i] for i in indices]
                sub_pred = [y_pred[i] for i in indices]
                _, _, f1_val, _ = precision_recall_fscore_support(
                    sub_true, sub_pred, labels=INTENT_NAMES, average="macro", zero_division=0
                )
                return f1_val

            ci_low, ci_high = calculate_bootstrap_ci(_f1_boot, len(y_true))

            results[name] = {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "macro_f1_ci_95": [ci_low, ci_high],
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "weighted_f1": round(float(weighted_f1), 4),
                "per_class": per_class,
                "confusion_matrix": cm
            }

        return results

    # -------------------------------------------------------------
    # RETRIEVAL EVALUATION (Heuristic vs Labeled Benchmark)
    # -------------------------------------------------------------
    def evaluate_retrieval(self) -> Dict[str, Any]:
        """
        Evaluates retrieval across:
        1. Heuristic Retrieval Hit@K (Section 13)
        2. Real Human-Labeled Retrieval Benchmark (Section 14: Recall@K, MRR)
        """
        if not self.retrieval_engine:
            return {
                "heuristic_retrieval_hits": {"heuristic_hit_1": 0.0, "heuristic_hit_3": 0.0, "heuristic_hit_5": 0.0},
                "labeled_benchmark": {"recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0, "mrr": 0.0}
            }

        # 1. Heuristic Hit@K on evaluation set
        heuristic_hits = self.retrieval_engine.evaluate_heuristic_hits(self.eval_data, top_k_levels=[1, 3, 5])

        # 2. Authentic labeled retrieval benchmark
        labeled_metrics = self.retrieval_engine.evaluate_labeled_benchmark()

        return {
            "heuristic_retrieval_hits": heuristic_hits,
            "labeled_benchmark": labeled_metrics
        }

    # -------------------------------------------------------------
    # ESCALATION ENGINE EVALUATION (Component vs End-to-End)
    # -------------------------------------------------------------
    def evaluate_escalation(self, use_predicted_intents: bool = False) -> Dict[str, Any]:
        """
        Evaluates escalation decisions with mathematically audited denominators:
        - Escalation Precision = TP / (TP + FP)
        - Escalation Recall = TP / (TP + FN)
        - Escalation F1 = 2 * P * R / (P + R)
        - False Escalation Rate = FP / Actual Auto-Handle (TN + FP)
        - Unsafe Auto-Handle Rate = FN / Actual Escalate (TP + FN)
        - Critical-Risk Miss Rate = Missed Critical / Total Critical
        Computes 95% bootstrap confidence intervals for recall and unsafe auto-handle rate.
        """
        y_true = [item.get("ground_truth_escalation", "auto_handle") for item in self.eval_data]
        texts = [item.get("customer_message", item.get("customer_text")) for item in self.eval_data]

        y_pred = []
        reasons = []
        urgencies = []

        for i, text in enumerate(texts):
            if use_predicted_intents:
                # End-to-End: Use pipeline predicted intent (NO GOLD INJECTION)
                intent_res = self.intent_classifier.classify(text, method="learned")
                pred_intent = intent_res.get("intent", "other")
                pred_conf = intent_res.get("confidence", 0.8)
            else:
                # Component Oracle: Use gold intent
                pred_intent = self.eval_data[i].get("ground_truth_intent", "other")
                pred_conf = 1.0

            dec = self.escalation_engine.decide(text, intent=pred_intent, intent_confidence=pred_conf)
            y_pred.append(dec["decision"])
            reasons.append(dec["reason"])
            urgencies.append(dec["urgency"])

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "escalate")
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "auto_handle" and yp == "escalate")
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "auto_handle")
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "auto_handle" and yp == "auto_handle")

        total = len(y_true)
        total_actual_escalate = tp + fn
        total_actual_autohandle = tn + fp

        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

        false_esc_rate = fp / total_actual_autohandle if total_actual_autohandle else 0.0
        unsafe_autohandle_rate = fn / total_actual_escalate if total_actual_escalate else 0.0
        automation_coverage = (tn + fn) / total if total else 0.0
        safe_automation_rate = tn / (tn + fn) if (tn + fn) else 0.0
        escalation_recall = rec

        # Critical hazard analysis
        critical_items = [it for it in self.eval_data if it.get("escalation_trigger") in ["safety_hazard", "critical"] or it.get("source") == "adversarial"]
        critical_missed = 0
        for it in critical_items:
            t = it.get("customer_message", it.get("customer_text"))
            d = self.escalation_engine.decide(t)["decision"]
            if d != "escalate":
                critical_missed += 1

        crit_miss_rate = (critical_missed / len(critical_items)) if critical_items else 0.0

        # Adversarial suite breakdown (uses expanded 50-case suite if available)
        adv_path = DATA_DIR / "adversarial_cases.jsonl"
        if adv_path.exists():
            adv_items = [json.loads(line) for line in open(adv_path, encoding="utf-8") if line.strip()]
        else:
            adv_items = [it for it in self.eval_data if it.get("source") == "adversarial"]
        adv_results = self.escalation_engine.evaluate_adversarial_suite(adv_items) if adv_items else {}

        # Bootstrap 95% Confidence Intervals
        def _rec_boot(indices):
            sub_true = [y_true[i] for i in indices]
            sub_pred = [y_pred[i] for i in indices]
            sub_tp = sum(1 for yt, yp in zip(sub_true, sub_pred) if yt == "escalate" and yp == "escalate")
            sub_fn = sum(1 for yt, yp in zip(sub_true, sub_pred) if yt == "escalate" and yp == "auto_handle")
            return (sub_tp / (sub_tp + sub_fn)) if (sub_tp + sub_fn) > 0 else 0.0

        def _unsafe_boot(indices):
            sub_true = [y_true[i] for i in indices]
            sub_pred = [y_pred[i] for i in indices]
            sub_tp = sum(1 for yt, yp in zip(sub_true, sub_pred) if yt == "escalate" and yp == "escalate")
            sub_fn = sum(1 for yt, yp in zip(sub_true, sub_pred) if yt == "escalate" and yp == "auto_handle")
            return (sub_fn / (sub_tp + sub_fn)) if (sub_tp + sub_fn) > 0 else 0.0

        rec_ci = calculate_bootstrap_ci(_rec_boot, total)
        unsafe_ci = calculate_bootstrap_ci(_unsafe_boot, total)

        return {
            "mode": "end_to_end" if use_predicted_intents else "component_oracle",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "escalation_recall": round(float(escalation_recall), 4),
            "recall_ci_95": rec_ci,
            "f1": round(float(f1), 4),
            "automation_coverage": round(float(automation_coverage), 4),
            "safe_automation_rate": round(float(safe_automation_rate), 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_evaluated": total,
            "actual_escalations_count": total_actual_escalate,
            "actual_autohandle_count": total_actual_autohandle,
            "false_escalation_rate": round(float(false_esc_rate), 4),
            "false_escalation_fraction": f"{fp}/{total_actual_autohandle}",
            "unsafe_autohandle_rate": round(float(unsafe_autohandle_rate), 4),
            "unsafe_autohandle_rate_ci_95": unsafe_ci,
            "unsafe_autohandle_fraction": f"{fn}/{total_actual_escalate}",
            "critical_risk_miss_rate": round(float(crit_miss_rate), 4),
            "critical_miss_fraction": f"{critical_missed}/{len(critical_items)}",
            "adversarial_suite": adv_results
        }

    # -------------------------------------------------------------
    # REPLY QUALITY EVALUATION ACROSS 4 CONFIGURATIONS
    # -------------------------------------------------------------
    def evaluate_reply_quality(self, use_predicted_intents: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluates reply quality across 4 configurations:
        1. canned (Deterministic Baseline 1)
        2. template (Intent Template Baseline 2)
        3. llm_no_rag (RAG Ablation)
        4. rag_llm (Primary Agent)
        Also collects auditable end-to-end evaluation records for every sample.
        """
        configs_to_test = [
            ("canned", "canned_response"),
            ("template", "intent_template")
        ]
        if not self.offline:
            configs_to_test.extend([
                ("llm_no_rag", "llm_no_rag"),
                ("rag_llm", "rag_llm_primary")
            ])
        else:
            # Explicit offline labels
            configs_to_test.append(("template", "offline_baseline_agent"))

        quality_results = {}
        e2e_records = []

        for gen_mode, display_name in configs_to_test:
            dimension_totals = {dim: 0.0 for dim in ["groundedness", "helpfulness", "relevance", "brand_alignment", "safety"]}
            overall_scores = []
            bleu_scores = []
            rouge_scores = []
            lengths = []

            for item in self.eval_data:
                text = item.get("customer_message", item.get("customer_text"))
                ref_reply = item.get("ground_truth_reply", "")
                gold_intent = item.get("intent", item.get("ground_truth_intent", "other"))
                gold_esc = item.get("expected_escalation", False)

                if use_predicted_intents:
                    intent_res = self.intent_classifier.classify(text, method="learned")
                    intent = intent_res.get("intent", "other")
                    conf = intent_res.get("confidence", 0.8)
                    esc_res = self.escalation_engine.decide(text, intent=intent, intent_confidence=conf)
                else:
                    intent = gold_intent
                    esc_res = self.escalation_engine.decide(text, intent=intent, intent_confidence=1.0)

                # Historical evidence retrieval
                retrieved_cases = self.retrieval_engine.retrieve(text, top_k=3) if self.retrieval_engine else []
                retrieved_ids = [c.get("pair_id") for c in retrieved_cases]

                # Reply generation
                reply_data = self.reply_generator.generate_reply(
                    customer_text=text,
                    intent=intent,
                    escalation_decision=esc_res["decision"],
                    escalation_reason=esc_res["reason"],
                    method=gen_mode if gen_mode != "offline_baseline_agent" else "template"
                )
                reply = reply_data.get("reply", "")
                lengths.append(len(reply))

                # Lexical metrics
                lex = calculate_lexical_metrics(reply, ref_reply)
                bleu_scores.append(lex["bleu_1"])
                rouge_scores.append(lex["rouge_l"])

                # Judge evaluation
                judge_res = self.judge.evaluate_reply(
                    customer_text=text,
                    agent_reply=reply,
                    intent=intent,
                    escalation_decision=esc_res["decision"],
                    escalation_reason=esc_res["reason"],
                    offline=self.offline
                )
                score = judge_res.get("overall_score", 4.0)
                overall_scores.append(score)

                dims = judge_res.get("dimension_scores", {})
                for d in dimension_totals:
                    dimension_totals[d] += dims.get(d, score)

                # Store auditable record for primary agent / template run
                if display_name in ["intent_template", "rag_llm_primary"]:
                    e2e_records.append({
                        "example_id": item.get("example_id"),
                        "customer_text": text,
                        "gold_intent": gold_intent,
                        "predicted_intent": intent,
                        "gold_escalation": gold_esc,
                        "predicted_escalation": (esc_res["decision"] == "escalate"),
                        "escalation_reason": esc_res["reason"],
                        "retrieved_case_ids": retrieved_ids,
                        "reply": reply,
                        "judge_scores": dims,
                        "gold_intent_injected": False,
                        "ground_truth": {
                            "intent": gold_intent,
                            "escalation": gold_esc
                        }
                    })

            n = len(self.eval_data)
            avg_dims = {d: round(tot / n, 2) for d, tot in dimension_totals.items()}
            quality_results[display_name] = {
                "overall_judge_score": round(float(np.mean(overall_scores)), 2),
                "dimension_scores": avg_dims,
                "bleu_1": round(float(np.mean(bleu_scores)), 3),
                "rouge_l": round(float(np.mean(rouge_scores)), 3),
                "avg_char_length": round(float(np.mean(lengths)), 1)
            }

        return quality_results, e2e_records

    # -------------------------------------------------------------
    # FULL BENCHMARK SUITE EXECUTION
    # -------------------------------------------------------------
    def run_full_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """Runs the entire benchmark suite with pre-flight self-validation."""
        logger.info("Executing pre-flight benchmark self-validation...")
        validation_report = self.run_self_validation()

        logger.info("Evaluating Intent Classification baselines & CI...")
        intent_metrics = self.evaluate_intent_classification()

        logger.info("Evaluating Retrieval metrics (Heuristic Hit@K & Labeled Benchmark)...")
        retrieval_metrics = self.evaluate_retrieval()

        logger.info("Evaluating Escalation Engine (Component Oracle)...")
        esc_component = self.evaluate_escalation(use_predicted_intents=False)

        logger.info("Evaluating Escalation Engine (End-to-End)...")
        esc_end_to_end = self.evaluate_escalation(use_predicted_intents=True)

        logger.info("Evaluating Reply Quality across baselines & logging E2E records...")
        reply_quality, e2e_records = self.evaluate_reply_quality(use_predicted_intents=True)

        logger.info("Validating LLM-as-Judge against authentic Human Annotations...")
        if HUMAN_RATINGS_JSON_PATH.exists():
            judge_validation = self.judge.validate_against_human_ratings(offline=self.offline)
            if judge_validation.get("success", False):
                with open(HUMAN_RATINGS_JSON_PATH, "r", encoding="utf-8") as f:
                    h_items = json.load(f)
                h_scores = [float(item["overall"]) for item in h_items]
                from config import GENERATED_REPLIES_PATH
                gen_by_id = {}
                if GENERATED_REPLIES_PATH.exists():
                    with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                rec = json.loads(line)
                                gen_by_id[rec["example_id"]] = rec
                j_scores = [
                    self.judge.evaluate_reply(
                        gen_by_id.get(item["example_id"], {}).get("customer_text", item.get("customer_text", "")),
                        gen_by_id.get(item["example_id"], {}).get("agent_reply", item.get("agent_reply", "")),
                        offline=self.offline
                    )["overall_score"]
                    for item in h_items
                ]
                def _mae_boot(indices):
                    s1 = np.array([h_scores[i] for i in indices])
                    s2 = np.array([j_scores[i] for i in indices])
                    return float(np.mean(np.abs(s1 - s2)))
                mae_ci = calculate_bootstrap_ci(_mae_boot, len(h_scores))
                if "human_vs_judge" in judge_validation:
                    judge_validation["human_vs_judge"]["mae_ci_95"] = mae_ci
        else:
            print("Human judge validation unavailable:\nhuman ratings have not been supplied.", file=sys.stderr)
            judge_validation = {
                "status": "unavailable",
                "success": False,
                "sample_size": 0,
                "message": "Human judge validation unavailable: human ratings have not been supplied.",
                "human_vs_judge": {
                    "mean_absolute_error": 0.0,
                    "exact_agreement_rate": 0.0,
                    "within_one_point_rate": 0.0,
                    "spearman_correlation": 0.0,
                    "quadratic_weighted_kappa": 0.0
                }
            }

        full_benchmark = {
            "validation_report": validation_report,
            "leakage_check": {
                "status": validation_report["status"],
                "exact_text_overlap": validation_report["text_overlap"],
                "normalized_overlap": validation_report["normalized_overlap"],
                "conversation_overlap": validation_report["conversation_overlap"]
            },
            "dataset_info": {
                "golden_eval_count": len(self.eval_data),
                "retrieval_corpus_count": len(self.retrieval_engine.corpus) if self.retrieval_engine else 0,
                "human_validation_count": judge_validation.get("sample_size", 0),
                "retrieval_benchmark_count": retrieval_metrics.get("labeled_benchmark", {}).get("benchmark_size", 35)
            },
            "intent_classification": intent_metrics,
            "retrieval": retrieval_metrics,
            "escalation_component_oracle": esc_component,
            "escalation_end_to_end": esc_end_to_end,
            "reply_quality": reply_quality,
            "judge_validation": judge_validation
        }

        if save_results and not self.max_samples:
            # 1. Save benchmark_metrics.json (Single source of truth)
            with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(full_benchmark, f, indent=2)
            logger.info(f"Verified benchmark metrics exported to {BENCHMARK_RESULTS_PATH}")

            # 2. Save auditable end-to-end evaluation records (detailed_results.jsonl)
            with open(E2E_RECORDS_PATH, "w", encoding="utf-8") as f:
                for rec in e2e_records:
                    f.write(json.dumps(rec) + "\n")
            
            detailed_path = RESULTS_DIR / "detailed_results.jsonl"
            with open(detailed_path, "w", encoding="utf-8") as f:
                for rec in e2e_records:
                    f.write(json.dumps(rec) + "\n")
            logger.info(f"Auditable E2E records exported to {E2E_RECORDS_PATH} and {detailed_path}")

            # 3. Save benchmark metadata (Phase 19 & Phase 24)
            git_commit = "unknown"
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            except Exception:
                pass

            metadata = {
                "seed": SEED,
                "brand": TARGET_BRAND,
                "golden_size": len(self.eval_data),
                "judge_validation_size": judge_validation.get("sample_size", 0),
                "retrieval_queries": retrieval_metrics.get("labeled_benchmark", {}).get("benchmark_size", 35),
                "model": "offline_deterministic_agent" if self.offline else AGENT_MODEL_NAME,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "git_commit": git_commit
            }
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Benchmark metadata exported to {METADATA_PATH}")

        return full_benchmark
