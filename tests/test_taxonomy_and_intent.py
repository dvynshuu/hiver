import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import INTENT_TAXONOMY, INTENT_NAMES
from src.intent_classifier import IntentClassifier

class TestTaxonomyAndIntent(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier()

    def test_taxonomy_completeness(self):
        """Verify all 8 intents contain definition, positive examples, boundary conditions, and confusable intents."""
        self.assertEqual(len(INTENT_NAMES), 8)
        expected_intents = {
            "device_issue", "software_bug", "account_security", "connectivity",
            "billing_purchase", "product_inquiry", "general_feedback", "other"
        }
        self.assertEqual(set(INTENT_NAMES), expected_intents)

        for name, meta in INTENT_TAXONOMY.items():
            self.assertIn("name", meta, f"Missing 'name' in intent {name}")
            self.assertIn("definition", meta, f"Missing 'definition' in intent {name}")
            self.assertIn("positive_examples", meta, f"Missing 'positive_examples' in intent {name}")
            self.assertIn("boundary_conditions", meta, f"Missing 'boundary_conditions' in intent {name}")
            self.assertIn("confusable_intents", meta, f"Missing 'confusable_intents' in intent {name}")
            self.assertGreaterEqual(len(meta["positive_examples"]), 3)

    def test_deterministic_majority_baseline(self):
        """Verify majority-class baseline is deterministic and uses zero random choice."""
        res1 = self.classifier.classify("My iPhone battery is dead", method="majority")
        res2 = self.classifier.classify("My iPhone battery is dead", method="majority")
        self.assertEqual(res1["intent"], "software_bug")
        self.assertEqual(res1["intent"], res2["intent"])
        self.assertEqual(res1["method"], "majority_class_baseline")

    def test_learned_baseline_classification(self):
        """Verify TF-IDF + Logistic Regression learned baseline outputs valid intents and confidence."""
        res_device = self.classifier.classify("My phone battery dies within 1 hour and screen flickers", method="learned")
        self.assertIn(res_device["intent"], INTENT_NAMES)
        self.assertGreater(res_device["confidence"], 0.0)
        self.assertLessEqual(res_device["confidence"], 1.0)
        self.assertEqual(res_device["method"], "tfidf_logistic_regression")

        res_sec = self.classifier.classify("Someone hacked my Apple ID password and locked my iCloud", method="learned")
        self.assertEqual(res_sec["intent"], "account_security")

    def test_empty_input_handling(self):
        """Verify classifier handles empty or whitespace input safely."""
        res = self.classifier.classify("", method="learned")
        self.assertEqual(res["intent"], "other")

if __name__ == "__main__":
    unittest.main()
