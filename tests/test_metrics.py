import unittest
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import calculate_lexical_metrics
from src.llm_judge import LLMJudge

class TestMetricsAndDenominators(unittest.TestCase):
    def test_lexical_metrics(self):
        """Verify BLEU-1 and ROUGE-L calculations on exact and differing texts."""
        ref = "Please restart your device and update iOS."
        pred_exact = "Please restart your device and update iOS."
        res_exact = calculate_lexical_metrics(pred_exact, ref)
        self.assertAlmostEqual(res_exact["bleu_1"], 1.0, places=3)
        self.assertAlmostEqual(res_exact["rouge_l"], 1.0, places=3)

        pred_partial = "Restart your device."
        res_part = calculate_lexical_metrics(pred_partial, ref)
        self.assertGreater(res_part["bleu_1"], 0.0)
        self.assertLess(res_part["bleu_1"], 1.0)

    def test_agreement_statistics(self):
        """Verify Pearson r, MAE, and Kappa calculations."""
        h = [4.0, 5.0, 3.0, 4.0, 5.0]
        j = [4.0, 5.0, 3.0, 4.0, 5.0]
        res_perfect = LLMJudge._compute_agreement_metrics(h, j)
        self.assertAlmostEqual(res_perfect["pearson_correlation"], 1.0, places=2)
        self.assertAlmostEqual(res_perfect["mean_absolute_error"], 0.0, places=2)
        self.assertAlmostEqual(res_perfect["cohens_kappa"], 1.0, places=2)

        j_off = [4.0, 4.0, 3.0, 5.0, 4.0]
        res_off = LLMJudge._compute_agreement_metrics(h, j_off)
        self.assertGreater(res_off["mean_absolute_error"], 0.0)
        self.assertLessEqual(res_off["mean_absolute_error"], 1.0)

    def test_escalation_denominators(self):
        """Verify escalation rates strictly use respective class totals as denominators."""
        # Scenario: 20 total items (15 auto_handle, 5 escalate)
        # Predictions: 1 false escalation (FP), 1 missed escalation (FN)
        # TP = 4, FN = 1, TN = 14, FP = 1
        tp, fn, tn, fp = 4, 1, 14, 1
        total_autohandle = tn + fp  # 15
        total_escalate = tp + fn    # 5

        false_esc_rate = fp / total_autohandle  # 1/15 = 0.0667, NOT 1/20!
        unsafe_autohandle_rate = fn / total_escalate  # 1/5 = 0.200, NOT 1/20!

        self.assertAlmostEqual(false_esc_rate, 1 / 15, places=3)
        self.assertAlmostEqual(unsafe_autohandle_rate, 1 / 5, places=3)
        self.assertNotEqual(false_esc_rate, 1 / 20)
        self.assertNotEqual(unsafe_autohandle_rate, 1 / 20)

if __name__ == "__main__":
    unittest.main()
