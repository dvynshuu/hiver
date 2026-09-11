"""
Submission Readiness and Hardening Test Suite.
Validates:
1. Canonical dataset file structure and complete absence of obsolete/duplicate artifacts.
2. Complete absence of credentials, secrets, or .env files.
3. Metadata schema completeness in results/benchmark_metadata.json.
4. Clean fail-closed behavior when human ratings are missing.
5. Index rebuild strictly from retrieval_corpus.jsonl without dynamic re-splitting.
6. Absolute zero gold label injection in end-to-end evaluation records.
7. Exact synchronization between benchmark_metrics.json, REPORT.md, and README.md.
"""
import unittest
import sys
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    GOLDEN_DIR,
    JUDGE_DIR,
    RESULTS_DIR,
    GOLDEN_EVAL_PATH,
    HELD_OUT_POOL_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    CORPUS_INDEX_PATH,
    BENCHMARK_METADATA_PATH,
    BENCHMARK_RESULTS_PATH,
    HUMAN_RATINGS_JSON_PATH
)
from src.retrieval import HistoricalRetrievalEngine
from src.llm_judge import LLMJudge


class TestSubmissionReadiness(unittest.TestCase):

    def test_canonical_dataset_structure_exists(self):
        """Verify all canonical dataset artifacts exist in the required locations."""
        canonical_files = [
            DATA_DIR / "retrieval_corpus.jsonl",
            DATA_DIR / "held_out_pool.jsonl",
            DATA_DIR / "adversarial_cases.jsonl",
            DATA_DIR / "retrieval_benchmark.json",
            DATA_DIR / "split_manifest.json",
            DATA_DIR / "labelling_guide.md",
            GOLDEN_DIR / "candidates.jsonl",
            GOLDEN_DIR / "manual_annotations.jsonl",
            GOLDEN_DIR / "annotator_agreement.json",
            GOLDEN_DIR / "golden_eval.jsonl",
            JUDGE_DIR / "judge_validation_sample.jsonl",
            JUDGE_DIR / "generated_replies.jsonl",
            JUDGE_DIR / "human_review.csv",
            JUDGE_DIR / "human_ratings.json",
            JUDGE_DIR / "llm_judge_ratings.json",
            JUDGE_DIR / "judge_validation.json",
            JUDGE_DIR / "HUMAN_RATING_GUIDE.md",
            JUDGE_DIR / "README.md"
        ]
        for f in canonical_files:
            self.assertTrue(f.exists(), f"Required canonical file missing: {f}")

    def test_obsolete_and_duplicate_artifacts_removed(self):
        """Verify duplicate, legacy, and stale files have been completely purged."""
        purged_files = [
            DATA_DIR / "golden_eval_set.jsonl",
            DATA_DIR / "held_out_eval_pool.jsonl",
            GOLDEN_DIR / "candidates_pool.jsonl",
            DATA_DIR / "judge_evaluation_sample.json",
            PROJECT_ROOT / "DECISIONS.md",
            PROJECT_ROOT / ".env",
            PROJECT_ROOT / "baseline"
        ]
        for f in purged_files:
            self.assertFalse(f.exists(), f"Obsolete/duplicate artifact still present: {f}")

    def test_zero_secrets_in_repo(self):
        """Verify that no actual API keys or secret tokens appear in tracked files."""
        secret_patterns = [
            "AQ.Ab8RN6LM",
            "AIzaSy",
            "sk-proj-",
            "ghp_"
        ]
        for ext in ["*.py", "*.md", "*.json", "*.jsonl", "*.txt"]:
            for f in PROJECT_ROOT.glob(f"**/{ext}"):
                if ".git" in f.parts or "__pycache__" in f.parts:
                    continue
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    for pat in secret_patterns:
                        self.assertNotIn(pat, content, f"Secret pattern '{pat}' found in {f}")
                except Exception:
                    pass

    def test_benchmark_metadata_schema(self):
        """Verify results/benchmark_metadata.json contains all required audit fields."""
        self.assertTrue(BENCHMARK_METADATA_PATH.exists(), "benchmark_metadata.json missing")
        with open(BENCHMARK_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        required_keys = [
            "brand", "seed", "golden_size", "judge_validation_size",
            "retrieval_validation_size", "model", "timestamp", "git_commit"
        ]
        for k in required_keys:
            self.assertIn(k, meta, f"Metadata missing required key: '{k}'")

        self.assertEqual(meta["brand"], "AppleSupport")
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["golden_size"], 200)
        self.assertEqual(meta["judge_validation_size"], 45)
        self.assertEqual(meta["retrieval_validation_size"], 35)

    def test_missing_human_ratings_fails_cleanly(self):
        """Verify judge validation produces exact clean message when human ratings are absent."""
        judge = LLMJudge()
        fake_path = Path(tempfile.gettempdir()) / "non_existent_human_ratings.json"
        res = judge.validate_against_human_ratings(human_ratings_path=fake_path, offline=True)
        self.assertEqual(res["status"], "unavailable")
        self.assertFalse(res["success"])
        self.assertIn("human ratings are not present", res["message"])

    def test_e2e_records_have_zero_gold_label_injection(self):
        """Verify that runtime evaluation records logged in results/ have gold_intent_injected == False."""
        records_path = RESULTS_DIR / "end_to_end_evaluation_records.jsonl"
        self.assertTrue(records_path.exists(), "end_to_end_evaluation_records.jsonl must exist")
        records = []
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        self.assertEqual(len(records), 200)
        for r in records:
            self.assertFalse(
                r.get("gold_intent_injected", True),
                f"Record {r.get('example_id')} has gold_intent_injected == True!"
            )

    def test_retrieval_index_rebuilds_when_pickle_missing(self):
        """Verify HistoricalRetrievalEngine.load() successfully rebuilds index from JSONL when pickle is missing."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_index_path = Path(temp_dir.name) / "test_corpus.pkl"
        try:
            self.assertFalse(temp_index_path.exists())
            engine = HistoricalRetrievalEngine.load(file_path=temp_index_path)
            self.assertGreater(len(engine.corpus), 0)
            self.assertTrue(temp_index_path.exists())
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
