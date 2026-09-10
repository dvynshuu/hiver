import unittest
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TestReportConsistency(unittest.TestCase):
    def setUp(self):
        metrics_path = PROJECT_ROOT / "results" / "benchmark_metrics.json"
        self.assertTrue(metrics_path.exists(), "results/benchmark_metrics.json must exist")
        with open(metrics_path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

        self.report_path = PROJECT_ROOT / "REPORT.md"
        self.assertTrue(self.report_path.exists(), "REPORT.md must exist")
        with open(self.report_path, "r", encoding="utf-8") as f:
            self.report_content = f.read()

        self.readme_path = PROJECT_ROOT / "README.md"
        self.assertTrue(self.readme_path.exists(), "README.md must exist")
        with open(self.readme_path, "r", encoding="utf-8") as f:
            self.readme_content = f.read()

    def test_metrics_in_report_and_readme(self):
        """Verify that key headline metrics in benchmark_metrics.json are accurately reflected in REPORT.md and README.md."""
        macro_f1 = self.metrics["intent_classification"]["offline_classifier"]["macro_f1"]
        esc_recall = self.metrics["escalation_end_to_end"]["recall"]
        false_esc = self.metrics["escalation_end_to_end"]["false_escalation_rate"]
        unsafe_auto = self.metrics["escalation_end_to_end"]["unsafe_autohandle_rate"]
        labeled_rec3 = self.metrics["retrieval"]["labeled_benchmark"]["recall_3"]

        # Check macro-f1 formatted as 0.595
        macro_f1_str = f"{macro_f1:.3f}"
        self.assertIn(macro_f1_str, self.report_content, f"Macro F1 {macro_f1_str} missing in REPORT.md")
        self.assertIn(macro_f1_str, self.readme_content, f"Macro F1 {macro_f1_str} missing in README.md")

        # Check escalation recall 0.853
        esc_recall_str = f"{esc_recall:.3f}"
        self.assertIn(esc_recall_str, self.report_content, f"Escalation recall {esc_recall_str} missing in REPORT.md")
        self.assertIn(esc_recall_str, self.readme_content, f"Escalation recall {esc_recall_str} missing in README.md")

        # Check false escalation rate 24.7%
        false_esc_pct = f"{false_esc * 100:.1f}%"
        self.assertIn(false_esc_pct, self.report_content, f"False escalation rate {false_esc_pct} missing in REPORT.md")
        self.assertIn(false_esc_pct, self.readme_content, f"False escalation rate {false_esc_pct} missing in README.md")

        # Check unsafe auto-handle rate 14.7%
        unsafe_auto_pct = f"{unsafe_auto * 100:.1f}%"
        self.assertIn(unsafe_auto_pct, self.report_content, f"Unsafe auto-handle rate {unsafe_auto_pct} missing in REPORT.md")
        self.assertIn(unsafe_auto_pct, self.readme_content, f"Unsafe auto-handle rate {unsafe_auto_pct} missing in README.md")

        # Check labeled retrieval Recall@3
        rec3_str = f"{labeled_rec3:.3f}"
        self.assertIn(rec3_str, self.report_content, f"Labeled Recall@3 {rec3_str} missing in REPORT.md")
        self.assertIn(rec3_str, self.readme_content, f"Labeled Recall@3 {rec3_str} missing in README.md")

if __name__ == "__main__":
    unittest.main()
