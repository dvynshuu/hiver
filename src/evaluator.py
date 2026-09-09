import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    INTENT_NAMES,
    GOLDEN_EVAL_PATH,
    BENCHMARK_RESULTS_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    HUMAN_EVAL_RATINGS_PATH,
    RESULTS_DIR
)
from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetrievalEngine
from src.reply_generator import ReplyGenerator
from src.escalation_engine import EscalationEngine
from src.llm_judge import LLMJudge
from src.data_pipeline import check_evaluation_leakage, format_leakage_report

logger = logging.getLogger(__name__)

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
    Comprehensive evaluation harness executing the Golden Evaluation Set.
    Separates:
    - Component-Level Evaluation (Oracle gold inputs)
    - End-to-End Evaluation (Pipeline predictions only, NO gold intent injection)
    Audits metric denominators, baseline comparisons, and leakage checks.
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
        self.retrieval_engine = retrieval_engine
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

    def run_leakage_check(self) -> Dict[str, Any]:
        """Validates that zero evaluation items exist in the retrieval corpus."""
        retrieval_corpus = []
        if RETRIEVAL_CORPUS_JSONL_PATH.exists():
            with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        retrieval_corpus.append(json.loads(line))
        elif self.retrieval_engine and self.retrieval_engine.corpus:
            retrieval_corpus = self.retrieval_engine.corpus

        leakage_results = check_evaluation_leakage(retrieval_corpus, self.eval_data)
        if leakage_results["status"] != "PASS":
            raise ValueError(f"CRITICAL: Evaluation Leakage Detected!\n{format_leakage_report(leakage_results)}")
        return leakage_results

    # -------------------------------------------------------------
    # INTENT CLASSIFICATION EVALUATION
    # -------------------------------------------------------------
    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """
        Evaluates intent classification across:
        1. Majority Class Baseline (Deterministic)
        2. TF-IDF + Logistic Regression Baseline (Learned)
        3. LLM / Primary Agent (Few-Shot Gemini)
        """
        y_true = [item.get("ground_truth_intent", item.get("intent")) for item in self.eval_data]
        texts = [item.get("customer_message", item.get("customer_text")) for item in self.eval_data]

        modes = [
            ("majority", "majority_baseline"),
            ("learned", "tfidf_lr_baseline")
        ]
        if not self.offline:
            modes.append(("llm", "primary_agent_llm"))

        results = {}
        for mode, name in modes:
            y_pred = []
            for t in texts:
                res = self.intent_classifier.classify(t, method=mode)
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
                    "support": int(cs[i])
                }
                for i, name_i in enumerate(INTENT_NAMES)
            }

            cm = confusion_matrix(y_true, y_pred, labels=INTENT_NAMES).tolist()

            results[name] = {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(macro_f1), 4),
                "macro_precision": round(float(macro_p), 4),
                "macro_recall": round(float(macro_r), 4),
                "weighted_f1": round(float(weighted_f1), 4),
                "per_class": per_class,
                "confusion_matrix": cm
            }

        return results

    # -------------------------------------------------------------
    # RETRIEVAL EVALUATION
    # -------------------------------------------------------------
    def evaluate_retrieval(self) -> Dict[str, float]:
        """Calculates Recall@1, Recall@3, and Recall@5 on evaluation queries."""
        if not self.retrieval_engine:
            return {"recall_1": 0.0, "recall_3": 0.0, "recall_5": 0.0}
        return self.retrieval_engine.evaluate_retrieval_metrics(self.eval_data, top_k_levels=[1, 3, 5])

    # -------------------------------------------------------------
    # ESCALATION ENGINE EVALUATION (WITH CORRECT DENOMINATORS)
    # -------------------------------------------------------------
    def evaluate_escalation(self, use_predicted_intents: bool = False) -> Dict[str, Any]:
        """
        Evaluates escalation decisions with mathematically rigorous denominators:
        - Escalation Precision = TP / (TP + FP)
        - Escalation Recall = TP / (TP + FN)
        - Escalation F1 = 2 * P * R / (P + R)
        - False Escalation Rate = FP / Actual Auto-Handle (TN + FP)
        - Unsafe Auto-Handle Rate = FN / Actual Escalate (TP + FN)
        - Critical-Risk Miss Rate = Missed Critical / Total Critical
        """
        y_true = [item.get("ground_truth_escalation", "auto_handle") for item in self.eval_data]
        texts = [item.get("customer_message", item.get("customer_text")) for item in self.eval_data]

        y_pred = []
        reasons = []
        urgencies = []

        for i, text in enumerate(texts):
            if use_predicted_intents:
                # End-to-End: Use predicted intent
                intent_res = self.intent_classifier.classify(text, method="learned")
                pred_intent = intent_res.get("intent", "other")
                pred_conf = intent_res.get("confidence", 0.8)
            else:
                # Component-Level: Use gold intent
                pred_intent = self.eval_data[i].get("ground_truth_intent", "other")
                pred_conf = 1.0

            dec = self.escalation_engine.decide(text, intent=pred_intent, intent_confidence=pred_conf)
            y_pred.append(dec["decision"])
            reasons.append(dec["reason"])
            urgencies.append(dec["urgency"])

        # Confusion matrix elements (escalate = positive, auto_handle = negative)
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

        # Critical hazard analysis (from adversarial items)
        critical_items = [it for it in self.eval_data if it.get("escalation_trigger") == "safety_hazard" or it.get("source") == "adversarial"]
        critical_missed = 0
        for it in critical_items:
            t = it.get("customer_message", it.get("customer_text"))
            d = self.escalation_engine.decide(t)["decision"]
            if d != "escalate":
                critical_missed += 1

        crit_miss_rate = (critical_missed / len(critical_items)) if critical_items else 0.0

        # Adversarial suite breakdown
        adv_items = [it for it in self.eval_data if it.get("source") == "adversarial"]
        adv_results = self.escalation_engine.evaluate_adversarial_suite(adv_items) if adv_items else {}

        return {
            "mode": "end_to_end" if use_predicted_intents else "component_oracle",
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
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
            "unsafe_autohandle_fraction": f"{fn}/{total_actual_escalate}",
            "critical_risk_miss_rate": round(float(crit_miss_rate), 4),
            "critical_miss_fraction": f"{critical_missed}/{len(critical_items)}",
            "adversarial_suite": adv_results
        }

    # -------------------------------------------------------------
    # REPLY QUALITY EVALUATION ACROSS 4 CONFIGURATIONS
    # -------------------------------------------------------------
    def evaluate_reply_quality(self, use_predicted_intents: bool = False) -> Dict[str, Any]:
        """
        Evaluates reply quality across the 4 ablation configurations:
        1. canned (Baseline 1)
        2. template (Baseline 2)
        3. llm_no_rag (RAG ablation)
        4. rag_llm (Primary Agent)
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

        quality_results = {}

        for gen_mode, display_name in configs_to_test:
            dimension_totals = {dim: 0.0 for dim in ["groundedness", "helpfulness", "relevance", "brand_alignment", "safety"]}
            overall_scores = []
            bleu_scores = []
            rouge_scores = []
            lengths = []

            for item in self.eval_data:
                text = item.get("customer_message", item.get("customer_text"))
                ref_reply = item.get("ground_truth_reply", "")

                if use_predicted_intents:
                    intent_res = self.intent_classifier.classify(text, method="learned")
                    intent = intent_res.get("intent", "other")
                    esc_res = self.escalation_engine.decide(text, intent=intent, intent_confidence=intent_res.get("confidence", 0.8))
                else:
                    intent = item.get("ground_truth_intent", "other")
                    esc_res = self.escalation_engine.decide(text, intent=intent, intent_confidence=1.0)

                # Generate reply
                reply_data = self.reply_generator.generate_reply(
                    customer_text=text,
                    intent=intent,
                    escalation_decision=esc_res["decision"],
                    escalation_reason=esc_res["reason"],
                    method=gen_mode
                )
                reply = reply_data.get("reply", "")
                lengths.append(len(reply))

                # Lexical metrics
                lex = calculate_lexical_metrics(reply, ref_reply)
                bleu_scores.append(lex["bleu_1"])
                rouge_scores.append(lex["rouge_l"])

                # Judge evaluation (heuristic if offline, LLM if live)
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

            n = len(self.eval_data)
            avg_dims = {d: round(tot / n, 2) for d, tot in dimension_totals.items()}
            quality_results[display_name] = {
                "overall_judge_score": round(float(np.mean(overall_scores)), 2),
                "dimension_scores": avg_dims,
                "bleu_1": round(float(np.mean(bleu_scores)), 3),
                "rouge_l": round(float(np.mean(rouge_scores)), 3),
                "avg_char_length": round(float(np.mean(lengths)), 1)
            }

        return quality_results

    # -------------------------------------------------------------
    # FULL BENCHMARK SUITE
    # -------------------------------------------------------------
    def run_full_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """Runs the entire benchmark suite with component vs end-to-end separation."""
        logger.info("Executing automated leakage check...")
        leakage = self.run_leakage_check()

        logger.info("Evaluating Intent Classification baselines...")
        intent_metrics = self.evaluate_intent_classification()

        logger.info("Evaluating Retrieval metrics...")
        retrieval_metrics = self.evaluate_retrieval()

        logger.info("Evaluating Escalation Engine (Component Oracle)...")
        esc_component = self.evaluate_escalation(use_predicted_intents=False)

        logger.info("Evaluating Escalation Engine (End-to-End)...")
        esc_end_to_end = self.evaluate_escalation(use_predicted_intents=True)

        logger.info("Evaluating Reply Quality across baselines and Primary Agent...")
        reply_quality = self.evaluate_reply_quality(use_predicted_intents=True)

        logger.info("Validating LLM-as-Judge against authentic Human Annotations...")
        judge_validation = self.judge.validate_against_human_ratings(offline=self.offline)

        full_benchmark = {
            "leakage_check": leakage,
            "dataset_info": {
                "golden_eval_count": len(self.eval_data),
                "retrieval_corpus_count": len(self.retrieval_engine.corpus) if self.retrieval_engine else 0,
                "human_validation_count": judge_validation.get("sample_size", 0)
            },
            "intent_classification": intent_metrics,
            "retrieval": retrieval_metrics,
            "escalation_component_oracle": esc_component,
            "escalation_end_to_end": esc_end_to_end,
            "reply_quality": reply_quality,
            "judge_validation": judge_validation
        }

        if save_results and not self.max_samples:
            with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(full_benchmark, f, indent=2)
            logger.info(f"Verified benchmark metrics successfully exported to {BENCHMARK_RESULTS_PATH}")

        return full_benchmark
