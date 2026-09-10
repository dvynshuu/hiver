import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import HistoricalRetrievalEngine
from config import RETRIEVAL_BENCHMARK_PATH

class TestLabeledRetrievalBenchmark(unittest.TestCase):
    def test_mrr_and_recall_calculation(self):
        """Verify MRR and Recall@K calculation on controlled mock predictions."""
        # 3 mock corpus items
        corpus = [
            {"pair_id": "doc_1", "customer_text": "iPhone battery dead", "support_reply": "Check settings"},
            {"pair_id": "doc_2", "customer_text": "MacBook Wi-Fi broken", "support_reply": "Reset router"},
            {"pair_id": "doc_3", "customer_text": "Apple ID locked", "support_reply": "Go to iforgot"}
        ]
        engine = HistoricalRetrievalEngine(corpus=corpus)

        # Mock benchmark: 2 queries
        # Query 1: relevant is doc_1 (retrieved at rank 1: RR = 1.0)
        # Query 2: relevant is doc_2 (retrieved at rank 1: RR = 1.0)
        benchmark = [
            {"query": "iPhone battery", "relevant_pair_ids": ["doc_1"]},
            {"query": "MacBook Wi-Fi", "relevant_pair_ids": ["doc_2"]}
        ]

        metrics = engine.evaluate_labeled_benchmark(benchmark)
        self.assertAlmostEqual(metrics["mrr"], 1.0, places=2)
        self.assertAlmostEqual(metrics["recall_at_1"], 1.0, places=2)
        self.assertAlmostEqual(metrics["recall_at_3"], 1.0, places=2)

    def test_actual_retrieval_benchmark_file(self):
        """Verify the actual retrieval benchmark file evaluates successfully and returns valid metrics."""
        self.assertTrue(RETRIEVAL_BENCHMARK_PATH.exists(), "retrieval_benchmark.json must exist")
        engine = HistoricalRetrievalEngine.load()
        metrics = engine.evaluate_labeled_benchmark()

        self.assertIn("mrr", metrics)
        self.assertIn("recall_at_1", metrics)
        self.assertIn("recall_at_3", metrics)
        self.assertIn("recall_at_5", metrics)
        self.assertGreater(metrics["benchmark_size"], 20)
        self.assertGreater(metrics["mrr"], 0.5)

if __name__ == "__main__":
    unittest.main()
