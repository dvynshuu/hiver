import unittest
import sys
import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GOLDEN_EVAL_PATH, INTENT_NAMES

class TestGoldenDatasetProvenance(unittest.TestCase):
    def setUp(self):
        self.golden_records = []
        with open(GOLDEN_EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.golden_records.append(json.loads(line))

    def test_golden_set_size_and_sources(self):
        """Verify the golden evaluation set contains exactly 200 items (180 held-out + 20 adversarial)."""
        self.assertEqual(len(self.golden_records), 200, "Golden eval set must contain exactly 200 items.")
        
        held_out_count = sum(1 for r in self.golden_records if r.get("source") == "twcs")
        adversarial_count = sum(1 for r in self.golden_records if r.get("source") == "adversarial")
        
        self.assertEqual(held_out_count, 180, "Must have exactly 180 held-out TWCS cases.")
        self.assertEqual(adversarial_count, 20, "Must have exactly 20 adversarial cases.")

    def test_annotator_provenance_fields(self):
        """Verify each record has honest human provenance metadata and no synthetic ground truth."""
        for r in self.golden_records:
            self.assertEqual(r.get("annotator_type"), "human_single_annotator", f"Record {r.get('example_id')} missing human provenance.")
            self.assertEqual(r.get("annotator"), "human", f"Record {r.get('example_id')} annotator not human.")
            self.assertEqual(r.get("split"), "held_out", f"Record {r.get('example_id')} split not held_out.")
            self.assertIn("example_id", r, "Record missing example_id.")
            self.assertIn("notes", r, f"Record {r.get('example_id')} missing audit notes.")

    def test_intent_coverage_and_stratification(self):
        """Verify all 8 intents in the taxonomy have at least 5 examples and zero zero-coverage intents."""
        intent_counts = Counter(r["intent"] for r in self.golden_records)
        
        for intent in INTENT_NAMES:
            self.assertIn(intent, intent_counts, f"Intent {intent} has 0 representation!")
            self.assertGreaterEqual(
                intent_counts[intent], 5,
                f"Intent {intent} has fewer than 5 examples ({intent_counts[intent]})."
            )

    def test_inter_annotator_agreement_file(self):
        """Verify provenance and annotator agreement documentation honestly declares single-annotator setup with zero simulated data."""
        agreement_path = PROJECT_ROOT / "data" / "golden" / "annotator_agreement.json"
        self.assertTrue(agreement_path.exists(), "annotator_agreement.json does not exist.")
        
        with open(agreement_path, "r", encoding="utf-8") as f:
            agreement = json.load(f)
            
        self.assertEqual(agreement.get("primary_annotator"), "human_single_annotator")
        self.assertFalse(agreement.get("simulated_data", True), "Must NOT contain simulated data!")
        self.assertIn("independently reviewed by one human annotator", agreement.get("statement", ""))

    def test_auditable_machine_suggestion_fields(self):
        """Verify auditable machine suggestion vs final human intent fields in manual annotations."""
        manual_path = PROJECT_ROOT / "data" / "golden" / "manual_annotations.jsonl"
        self.assertTrue(manual_path.exists(), "manual_annotations.jsonl must exist.")
        
        with open(manual_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.assertIn("machine_suggestion", item)
                    self.assertIn("final_intent", item)
                    self.assertIn("label_changed", item)
                    self.assertEqual(item.get("annotator_type"), "human_single_annotator")

if __name__ == "__main__":
    unittest.main()
