import unittest
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RETRIEVAL_CORPUS_JSONL_PATH, GOLDEN_EVAL_PATH
from src.data_pipeline import check_evaluation_leakage

class TestLeakageDetection(unittest.TestCase):
    def test_actual_golden_dataset_leakage(self):
        """Verify the actual golden evaluation set has ZERO leakage against the retrieval corpus."""
        retrieval_corpus = []
        with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    retrieval_corpus.append(json.loads(line))

        eval_set = []
        with open(GOLDEN_EVAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    eval_set.append(json.loads(line))

        result = check_evaluation_leakage(retrieval_corpus, eval_set)
        self.assertEqual(result["exact_text_overlap"], 0, "Exact text leakage detected!")
        self.assertEqual(result["normalized_overlap"], 0, "Normalized text leakage detected!")
        self.assertEqual(result["conversation_overlap"], 0, "Conversation ID leakage detected!")
        self.assertEqual(result["status"], "PASS")

    def test_leakage_detector_catches_synthetic_leakage(self):
        """Verify the leakage detector reliably catches simulated exact, normalized, and conversation leaks."""
        corpus = [
            {
                "customer_text": "My iPhone battery dies too quickly after iOS 11 update",
                "conversation_id": "conv_9999",
                "customer_tweet_id": "9999"
            }
        ]

        # 1. Exact leak
        eval_exact = [{"customer_text": "My iPhone battery dies too quickly after iOS 11 update", "conversation_id": "conv_1111"}]
        res_exact = check_evaluation_leakage(corpus, eval_exact)
        self.assertGreater(res_exact["exact_text_overlap"], 0)
        self.assertEqual(res_exact["status"], "FAIL")

        # 2. Normalized leak (differing punctuation and casing)
        eval_norm = [{"customer_text": "my iphone battery dies too quickly after ios 11 update!!!", "conversation_id": "conv_2222"}]
        res_norm = check_evaluation_leakage(corpus, eval_norm)
        self.assertGreater(res_norm["normalized_overlap"], 0)
        self.assertEqual(res_norm["status"], "FAIL")

        # 3. Conversation ID leak
        eval_conv = [{"customer_text": "Completely different query text", "conversation_id": "conv_9999"}]
        res_conv = check_evaluation_leakage(corpus, eval_conv)
        self.assertGreater(res_conv["conversation_overlap"], 0)
        self.assertEqual(res_conv["status"], "FAIL")

if __name__ == "__main__":
    unittest.main()
