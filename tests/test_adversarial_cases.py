import unittest
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation_engine import EscalationEngine

class TestAdversarialCases(unittest.TestCase):
    def setUp(self):
        self.engine = EscalationEngine()
        self.cases_path = PROJECT_ROOT / "data" / "adversarial_cases.jsonl"
        self.assertTrue(self.cases_path.exists(), "data/adversarial_cases.jsonl must exist")
        self.cases = []
        with open(self.cases_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.cases.append(json.loads(line))

    def test_adversarial_suite_50_cases_coverage(self):
        """Verify that 50 adversarial cases are tested and all 5 categories achieve 100% recall."""
        self.assertEqual(len(self.cases), 50, "Must have exactly 50 adversarial cases")
        metrics = self.engine.evaluate_adversarial_suite(self.cases)
        self.assertEqual(metrics["total_adversarial_tested"], 50)
        self.assertEqual(metrics["total_adversarial_caught"], 50)
        self.assertEqual(metrics["overall_critical_risk_recall"], 1.0)
        self.assertEqual(metrics["physical_safety_recall"], 1.0)
        self.assertEqual(metrics["security_recall"], 1.0)
        self.assertEqual(metrics["financial_recall"], 1.0)
        self.assertEqual(metrics["legal_recall"], 1.0)
        self.assertEqual(metrics["human_request_recall"], 1.0)

if __name__ == "__main__":
    unittest.main()
