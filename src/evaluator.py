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
    RESULTS_DIR
)
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator
from src.escalation_engine import EscalationEngine
from src.llm_judge import LLMJudge
from src.retrieval import HistoricalRetrievalEngine

logger = logging.getLogger(__name__)

def calculate_token_f1_rouge_bleu(prediction: str, reference: str) -> Dict[str, float]:
    """
    Computes lexical overlap metrics (BLEU-1, Token-F1 / ROUGE-1 approximate, ROUGE-L LCS approximate).
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"bleu_1": 0.0, "rouge_l": 0.0, "token_f1": 0.0}

    # Precision and recall of unigrams (BLEU-1)
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    pred_overlap = sum(1 for t in pred_tokens if t in common_tokens)
    ref_overlap = sum(1 for t in ref_tokens if t in common_tokens)

    p = pred_overlap / len(pred_tokens) if pred_tokens else 0.0
    r = ref_overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    # Longest Common Subsequence (ROUGE-L approximation)
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
    Comprehensive evaluation harness running the Golden Evaluation Set
    against Trivial Baseline, Simple Baseline, and Primary AI Agent.
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

        # Initialize engines
        self.intent_classifier = IntentClassifier()
        self.retrieval_engine = retrieval_engine
        self.reply_generator = ReplyGenerator(retrieval_engine=self.retrieval_engine)
        self.escalation_engine = EscalationEngine()
        self.judge = LLMJudge()

    def _load_eval_set(self):
        """Load Golden Evaluation Set from disk with optional stratified subsampling."""
        if not self.eval_set_path.exists():
            raise FileNotFoundError(f"Golden evaluation set not found at {self.eval_set_path}")
        
        all_items = []
        with open(self.eval_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_items.append(json.loads(line))

        if self.max_samples and len(all_items) > self.max_samples:
            from collections import defaultdict
            by_intent = defaultdict(list)
            for item in all_items:
                by_intent[item["ground_truth_intent"]].append(item)
            
            per_intent = max(1, self.max_samples // len(by_intent))
            sampled = []
            for intent, items in by_intent.items():
                sampled.extend(items[:per_intent])
            self.eval_data = sampled[:self.max_samples]
        else:
            self.eval_data = all_items

    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """
        Evaluate intent classification for:
        1. Trivial Baseline (Random)
        2. Simple Baseline (Keyword Matching)
        3. Primary Agent (Few-Shot LLM / Hybrid)
        """
        y_true = [item["ground_truth_intent"] for item in self.eval_data]
        texts = [item["customer_message"] for item in self.eval_data]

        results = {}

        modes_to_test = [
            ("trivial_baseline", "trivial_random"),
            ("simple_baseline", "simple_keyword"),
            ("simple_baseline" if self.offline else "llm", "primary_agent")
        ]

        for mode, method_name in modes_to_test:
            y_pred = []
            for text in texts:
                try:
                    res = self.intent_classifier.classify(text, method=mode)
                    y_pred.append(res.get("intent", "other"))
                except Exception as e:
                    logger.warning(f"Intent classification failed in {mode}: {e}. Falling back to 'other'.")
                    y_pred.append("other")

            acc = accuracy_score(y_true, y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=INTENT_NAMES, average="macro", zero_division=0
            )

            # Per-class breakdown
            class_prec, class_rec, class_f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=INTENT_NAMES, average=None, zero_division=0
            )
            per_class = {
                name: {
                    "precision": round(float(class_prec[i]), 3),
                    "recall": round(float(class_rec[i]), 3),
                    "f1": round(float(class_f1[i]), 3)
                }
                for i, name in enumerate(INTENT_NAMES)
            }

            cm = confusion_matrix(y_true, y_pred, labels=INTENT_NAMES).tolist()

            results[method_name] = {
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "macro_precision": round(float(prec), 4),
                "macro_recall": round(float(rec), 4),
                "per_class": per_class,
                "confusion_matrix": cm,
                "predictions": y_pred
            }

        return results

    def evaluate_escalation(self) -> Dict[str, Any]:
        """
        Evaluate Escalation Decision Engine against ground truth.
        """
        y_true = [item["ground_truth_escalation"] for item in self.eval_data]
        texts = [item["customer_message"] for item in self.eval_data]
        intents = [item["ground_truth_intent"] for item in self.eval_data]

        y_pred = []
        reasons = []

        for text, intent in zip(texts, intents):
            dec = self.escalation_engine.decide(text, intent=intent)
            y_pred.append(dec["decision"])
            reasons.append(dec["reason"])

        acc = accuracy_score(y_true, y_pred)
        # Escalation class specific metrics (escalate = positive)
        prec, rec, f1, _ = precision_recall_fscore_support(
            [1 if y == "escalate" else 0 for y in y_true],
            [1 if y == "escalate" else 0 for y in y_pred],
            average="binary",
            zero_division=0
        )

        # False escalation rate and Missed escalation rate
        total_auto = sum(1 for y in y_true if y == "auto_handle")
        total_esc = sum(1 for y in y_true if y == "escalate")

        false_escalations = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "auto_handle" and yp == "escalate")
        missed_escalations = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "escalate" and yp == "auto_handle")

        false_esc_rate = (false_escalations / total_auto) if total_auto else 0.0
        missed_esc_rate = (missed_escalations / total_esc) if total_esc else 0.0

        return {
            "accuracy": round(float(acc), 4),
            "escalation_precision": round(float(prec), 4),
            "escalation_recall": round(float(rec), 4),
            "escalation_f1": round(float(f1), 4),
            "false_escalation_rate": round(float(false_esc_rate), 4),
            "missed_escalation_rate": round(float(missed_esc_rate), 4),
            "false_escalations_count": false_escalations,
            "missed_escalations_count": missed_escalations,
            "total_evaluated": len(y_true)
        }

    def evaluate_reply_quality_and_judge(self) -> Dict[str, Any]:
        """
        Evaluate reply quality across 3 systems (Trivial, Simple, Primary Agent)
        using both lexical metrics (BLEU-1, ROUGE-L) and LLM-as-a-Judge (5 dimensions).
        Also calculates human agreement calibration.
        """
        eval_subset = self.eval_data  # evaluate all 200 items

        models_to_test = [
            ("trivial_random", "trivial_baseline"),
            ("simple_template", "simple_baseline"),
            ("primary_agent", "simple_baseline" if self.offline else "rag_llm")
        ]

        quality_results = {}
        human_ratings = []
        agent_judge_ratings = []

        for model_key, gen_mode in models_to_test:
            dimension_totals = {dim: 0.0 for dim in ["relevance", "helpfulness", "tone", "groundedness", "completeness"]}
            overall_scores = []
            bleu_scores = []
            rouge_scores = []
            lengths = []

            for i, item in enumerate(eval_subset):
                text = item["customer_message"]
                gt_intent = item["ground_truth_intent"]
                gt_reply = item["ground_truth_reply"]

                # Determine escalation context
                esc_data = self.escalation_engine.decide(text, intent=gt_intent)

                # Generate reply with fallback safety
                try:
                    reply_data = self.reply_generator.generate_reply(
                        text,
                        intent=gt_intent,
                        escalation_decision=esc_data["decision"],
                        escalation_reason=esc_data["reason"],
                        method=gen_mode
                    )
                except Exception as e:
                    logger.warning(f"Reply generation failed for item {i} ({model_key}): {e}. Using fallback.")
                    reply_data = self.reply_generator.generate_reply_simple_baseline(text, gt_intent, esc_data["decision"])

                reply = reply_data.get("reply", "")
                lengths.append(len(reply))

                # Lexical metrics vs ground truth
                lex = calculate_token_f1_rouge_bleu(reply, gt_reply)
                bleu_scores.append(lex["bleu_1"])
                rouge_scores.append(lex["rouge_l"])

                # Judge evaluation with fallback safety
                try:
                    if self.offline:
                        eval_res = self.judge._heuristic_fallback_judge(text, reply, gt_intent)
                    else:
                        eval_res = self.judge.evaluate_reply(
                            customer_text=text,
                            agent_reply=reply,
                            intent=gt_intent,
                            escalation_decision=esc_data["decision"],
                            escalation_reason=esc_data["reason"]
                        )
                except Exception as e:
                    logger.warning(f"Judge evaluation failed for item {i} ({model_key}): {e}. Using heuristic fallback.")
                    eval_res = self.judge._heuristic_fallback_judge(text, reply, gt_intent)

                score = eval_res.get("overall_score", 3.0)
                overall_scores.append(score)

                dim_scores = eval_res.get("dimension_scores", {})
                for dim in dimension_totals:
                    dimension_totals[dim] += dim_scores.get(dim, score)

                # Collect human calibration comparison pairs for Primary Agent
                if model_key == "primary_agent" and "human_judge_scores" in item:
                    h_scores = item["human_judge_scores"]
                    h_avg = sum(h_scores.values()) / len(h_scores)
                    human_ratings.append(h_avg)
                    agent_judge_ratings.append(score)

            n = len(eval_subset)
            dim_averages = {dim: round(tot / n, 2) for dim, tot in dimension_totals.items()}
            quality_results[model_key] = {
                "overall_judge_score": round(float(np.mean(overall_scores)), 2),
                "dimension_scores": dim_averages,
                "bleu_1": round(float(np.mean(bleu_scores)), 3),
                "rouge_l": round(float(np.mean(rouge_scores)), 3),
                "avg_char_length": round(float(np.mean(lengths)), 1)
            }

        # Human agreement calibration
        human_agreement = self.judge.calculate_human_agreement(human_ratings, agent_judge_ratings)

        return {
            "systems": quality_results,
            "human_judge_agreement": human_agreement
        }

    def run_full_benchmark(self, save_results: bool = True) -> Dict[str, Any]:
        """Runs the complete end-to-end benchmark suite."""
        logger.info("Executing Intent Classification benchmark...")
        intent_metrics = self.evaluate_intent_classification()

        logger.info("Executing Escalation Decision benchmark...")
        escalation_metrics = self.evaluate_escalation()

        logger.info("Executing Reply Quality & LLM-as-a-Judge benchmark...")
        reply_metrics = self.evaluate_reply_quality_and_judge()

        full_benchmark = {
            "intent_classification": intent_metrics,
            "escalation": escalation_metrics,
            "reply_quality": reply_metrics["systems"],
            "human_judge_agreement": reply_metrics["human_judge_agreement"]
        }

        # Save to disk
        if save_results and not self.max_samples:
            with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(full_benchmark, f, indent=2)
            logger.info(f"Full benchmark results successfully saved to {BENCHMARK_RESULTS_PATH}!")

        return full_benchmark
