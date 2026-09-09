import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import HistoricalRetrievalEngine

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.sample_corpus = [
            {
                "pair_id": "test_001",
                "customer_text": "My iPhone battery dies within 2 hours after updating iOS",
                "support_reply": "We'd like to help check your battery health in Settings > Battery."
            },
            {
                "pair_id": "test_002",
                "customer_text": "Wi-Fi keeps disconnecting on my MacBook Pro every few minutes",
                "support_reply": "Try renewing your DHCP lease in Network preferences or resetting your router."
            },
            {
                "pair_id": "test_003",
                "customer_text": "How do I reset my Apple ID password if I forgot my trusted number",
                "support_reply": "Visit iforgot.apple.com to initiate account recovery."
            }
        ]
        self.engine = HistoricalRetrievalEngine(corpus=self.sample_corpus)

    def test_retrieve_top_k(self):
        """Verify retrieval returns relevant matches ranked by similarity with evidence IDs."""
        results = self.engine.retrieve("battery drain issue on my iPhone", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 2)
        top_match = results[0]
        self.assertEqual(top_match["pair_id"], "test_001")
        self.assertIn("evidence_id", top_match)
        self.assertGreater(top_match["similarity_score"], 0.0)

    def test_empty_query(self):
        """Verify empty query returns empty list without exception."""
        results = self.engine.retrieve("", top_k=3)
        self.assertEqual(results, [])

    def test_context_formatting(self):
        """Verify format_retrieval_context formats retrieved cases with Evidence IDs."""
        retrieved = self.engine.retrieve("Wi-Fi network", top_k=1)
        context = self.engine.format_retrieval_context(retrieved)
        self.assertIn("Historical Case #1", context)
        self.assertIn("Evidence ID: test_002", context)

if __name__ == "__main__":
    unittest.main()
