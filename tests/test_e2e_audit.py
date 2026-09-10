import unittest
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TestEndToEndAudit(unittest.TestCase):
    def test_audit_records_integrity(self):
        """Verify the auditable end-to-end evaluation records contain NO gold label injection."""
        records_path = PROJECT_ROOT / "results" / "end_to_end_evaluation_records.jsonl"
        self.assertTrue(records_path.exists(), "end_to_end_evaluation_records.jsonl must exist.")

        records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        self.assertEqual(len(records), 200, "Should have 200 auditable evaluation records.")

        for r in records:
            # Crucial requirement: No gold label injection
            self.assertFalse(
                r.get("gold_intent_injected", True),
                f"Record {r.get('example_id')} has gold_intent_injected=True!"
            )
            self.assertIn("example_id", r)
            self.assertIn("customer_text", r)
            self.assertIn("predicted_intent", r)
            self.assertIn("predicted_escalation", r)
            self.assertIn("escalation_reason", r)
            self.assertIn("retrieved_case_ids", r)
            self.assertIn("reply", r)
            self.assertIn("judge_scores", r)

if __name__ == "__main__":
    unittest.main()
