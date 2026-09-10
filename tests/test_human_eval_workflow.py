"""
Unit tests for the Human-Evaluation Workflow & LLM-as-Judge Validation.

Ensures strict compliance with core requirements:
1. Fail-closed behavior when human ratings are missing.
2. Strict validation of 1-5 ordinal rubric, blank ratings, duplicates, unknown IDs.
3. ID consistency between samples, generated replies, review CSV, and ratings.
4. Exactly 45 diverse evaluation samples with matching IDs.
5. Absolute zero synthetic or simulated human ratings across the entire codebase.
6. Agent replies evaluated rather than historical gold tweets.
7. Zero gold-intent leakage into the production agent during judge sample generation.
8. Authentic human provenance (rater_type = 'human').
9. Honest inter-rater agreement reporting.
"""
import unittest
import sys
import json
import csv
import io
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_DIR,
    JUDGE_SAMPLE_PATH,
    GENERATED_REPLIES_PATH,
    HUMAN_REVIEW_CSV_PATH,
    HUMAN_RATINGS_JSON_PATH,
    LLM_JUDGE_RATINGS_PATH,
    JUDGE_VALIDATION_RESULTS_PATH,
    JUDGE_VALIDATION_DATA_PATH
)
from scripts.import_human_ratings import (
    parse_and_validate_ratings,
    RATING_DIMENSIONS,
    load_expected_ids
)
from scripts.evaluate_judge import compute_dimension_metrics


class TestHumanEvalWorkflow(unittest.TestCase):

    def setUp(self):
        self.expected_ids = load_expected_ids()
        self.sorted_ids = sorted(self.expected_ids)

    def _create_mock_csv(self, rows_dict):
        """Helper to create a temporary CSV file with given rows."""
        temp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8", newline="")
        fieldnames = ["example_id", "groundedness", "helpfulness", "relevance", "brand_alignment", "safety", "overall", "notes"]
        writer = csv.DictWriter(temp, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_dict:
            writer.writerow(r)
        temp.close()
        return Path(temp.name)

    def test_sample_count_requirement(self):
        """Verify that judge validation samples, replies, review CSV, and ratings each contain exactly 45 items."""
        # 1. judge_validation_sample.jsonl
        self.assertTrue(JUDGE_SAMPLE_PATH.exists(), f"Missing {JUDGE_SAMPLE_PATH}")
        with open(JUDGE_SAMPLE_PATH, "r", encoding="utf-8") as f:
            sample_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(sample_records), 45, f"Expected 45 judge samples, found {len(sample_records)}")

        # 2. generated_replies.jsonl
        self.assertTrue(GENERATED_REPLIES_PATH.exists(), f"Missing {GENERATED_REPLIES_PATH}")
        with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
            reply_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(reply_records), 45, f"Expected 45 generated replies, found {len(reply_records)}")

        # 3. human_review.csv
        self.assertTrue(HUMAN_REVIEW_CSV_PATH.exists(), f"Missing {HUMAN_REVIEW_CSV_PATH}")
        with open(HUMAN_REVIEW_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        self.assertEqual(len(csv_rows), 45, f"Expected 45 CSV rows, found {len(csv_rows)}")

        # 4. human_ratings.json
        self.assertTrue(HUMAN_RATINGS_JSON_PATH.exists(), f"Missing {HUMAN_RATINGS_JSON_PATH}")
        with open(HUMAN_RATINGS_JSON_PATH, "r", encoding="utf-8") as f:
            human_ratings = json.load(f)
        self.assertEqual(len(human_ratings), 45, f"Expected 45 human ratings, found {len(human_ratings)}")

        # 5. llm_judge_ratings.json
        self.assertTrue(LLM_JUDGE_RATINGS_PATH.exists(), f"Missing {LLM_JUDGE_RATINGS_PATH}")
        with open(LLM_JUDGE_RATINGS_PATH, "r", encoding="utf-8") as f:
            judge_ratings = json.load(f)
        self.assertEqual(len(judge_ratings), 45, f"Expected 45 judge ratings, found {len(judge_ratings)}")

        # Verify all 45 IDs match across all artifacts
        sample_ids = {r["example_id"] for r in sample_records}
        reply_ids = {r["example_id"] for r in reply_records}
        csv_ids = {r["example_id"] for r in csv_rows}
        human_ids = {r["example_id"] for r in human_ratings}
        judge_ids = {r["example_id"] for r in judge_ratings}

        self.assertEqual(sample_ids, reply_ids, "Sample IDs do not match generated reply IDs.")
        self.assertEqual(sample_ids, csv_ids, "Sample IDs do not match CSV review IDs.")
        self.assertEqual(sample_ids, human_ids, "Sample IDs do not match human ratings IDs.")
        self.assertEqual(sample_ids, judge_ids, "Sample IDs do not match judge ratings IDs.")

    def test_human_ratings_file_missing_fails_closed(self):
        """Verify that when human ratings or review CSV is missing, the validator raises FileNotFoundError."""
        non_existent_csv = BASE_DIR / "data" / "judge" / "non_existent_review.csv"
        with self.assertRaises(FileNotFoundError):
            parse_and_validate_ratings(non_existent_csv)

    def test_missing_blank_rating_raises_validation_error(self):
        """Verify that blank/empty ratings trigger a validation error."""
        rows = []
        for i, eid in enumerate(self.sorted_ids):
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            row["notes"] = "Valid note"
            if i == 5:
                row["helpfulness"] = ""  # Intentionally blank
            rows.append(row)

        temp_csv = self._create_mock_csv(rows)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv)
            self.assertIn("Blank rating", str(ctx.exception))
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    def test_out_of_range_rating_raises_validation_error(self):
        """Verify that ratings < 1, > 5, or non-integer trigger a validation error."""
        # 1. Rating < 1
        rows_low = []
        for i, eid in enumerate(self.sorted_ids):
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            if i == 0:
                row["overall"] = "0"
            rows_low.append(row)
        temp_csv_low = self._create_mock_csv(rows_low)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv_low)
            self.assertIn("out of range", str(ctx.exception))
        finally:
            if temp_csv_low.exists():
                temp_csv_low.unlink()

        # 2. Rating > 5
        rows_high = []
        for i, eid in enumerate(self.sorted_ids):
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            if i == 0:
                row["safety"] = "6"
            rows_high.append(row)
        temp_csv_high = self._create_mock_csv(rows_high)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv_high)
            self.assertIn("out of range", str(ctx.exception))
        finally:
            if temp_csv_high.exists():
                temp_csv_high.unlink()

        # 3. Non-integer rating
        rows_float = []
        for i, eid in enumerate(self.sorted_ids):
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            if i == 0:
                row["groundedness"] = "3.5"
            rows_float.append(row)
        temp_csv_float = self._create_mock_csv(rows_float)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv_float)
            self.assertIn("must be an integer", str(ctx.exception))
        finally:
            if temp_csv_float.exists():
                temp_csv_float.unlink()

    def test_duplicate_example_id_raises_validation_error(self):
        """Verify that duplicate example IDs in the review CSV trigger a validation error."""
        rows = []
        for eid in self.sorted_ids:
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            rows.append(row)
        # Duplicate the first row
        rows[1]["example_id"] = rows[0]["example_id"]

        temp_csv = self._create_mock_csv(rows)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv)
            self.assertIn("Duplicate example_id found", str(ctx.exception))
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    def test_unknown_or_unexpected_example_id_raises_validation_error(self):
        """Verify that an unknown example ID outside the 45 sample IDs triggers a validation error."""
        rows = []
        for eid in self.sorted_ids:
            row = {dim: "4" for dim in RATING_DIMENSIONS}
            row["example_id"] = eid
            rows.append(row)
        rows[0]["example_id"] = "totally_fake_id_999"

        temp_csv = self._create_mock_csv(rows)
        try:
            with self.assertRaises(ValueError) as ctx:
                parse_and_validate_ratings(temp_csv)
            self.assertIn("Unknown example_id", str(ctx.exception))
        finally:
            if temp_csv.exists():
                temp_csv.unlink()

    def test_every_record_in_human_ratings_has_rater_type_human(self):
        """Verify that every record in human_ratings.json has rater_type == 'human'."""
        with open(HUMAN_RATINGS_JSON_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)

        for r in records:
            self.assertEqual(r.get("rater_type"), "human", f"Record {r.get('example_id')} does not have rater_type == 'human'")
            self.assertEqual(r.get("rater_id"), "human_1")
            for dim in RATING_DIMENSIONS:
                val = r.get(dim)
                self.assertIsInstance(val, int, f"Dimension {dim} in {r.get('example_id')} is not an int.")
                self.assertGreaterEqual(val, 1)
                self.assertLessEqual(val, 5)

    def test_judge_evaluation_evaluates_agent_reply(self):
        """Verify that judge evaluation evaluates actual agent replies, not historical 2017 tweets."""
        with open(GENERATED_REPLIES_PATH, "r", encoding="utf-8") as f:
            gen_replies = [json.loads(line) for line in f if line.strip()]

        with open(LLM_JUDGE_RATINGS_PATH, "r", encoding="utf-8") as f:
            judge_ratings = json.load(f)

        judge_by_id = {r["example_id"]: r for r in judge_ratings}

        for item in gen_replies:
            eid = item["example_id"]
            self.assertIn("agent_reply", item)
            self.assertTrue(len(item["agent_reply"]) > 0, f"Empty agent reply for {eid}")
            # Ensure the judge received and recorded the agent's reply
            self.assertIn(eid, judge_by_id)
            self.assertEqual(
                item["agent_reply"],
                judge_by_id[eid]["agent_reply"],
                f"Judge did not evaluate the agent's generated reply for {eid}."
            )

    def test_gold_intent_never_passed_during_judge_sample_generation(self):
        """Verify that scripts/generate_judge_samples.py runs the production agent without gold intent injection."""
        script_path = PROJECT_ROOT / "scripts" / "generate_judge_samples.py"
        self.assertTrue(script_path.exists())
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Verify that classification, escalation, and reply generation do NOT use gold_intent
        self.assertIn("intent_classifier.classify(customer_text", code)
        self.assertIn("intent=pred_intent", code)
        self.assertNotIn("intent=gold_intent", code)
        self.assertNotIn("classify(customer_text, gold_intent", code)

    def test_no_synthetic_human_rating_generator_exists(self):
        """Audit all python scripts in scripts/ and src/ to ensure NO synthetic human ratings exist."""
        forbidden_patterns = [
            "rater_2 = rater_1",
            "rater_2 = copy",
            "r_grounded = 5",
            "fake_ratings",
            "simulated_human",
            "generate_fake_human",
            "mock_human_ratings"
        ]

        search_dirs = [PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"]
        for sdir in search_dirs:
            for pyfile in sdir.rglob("*.py"):
                with open(pyfile, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                for pattern in forbidden_patterns:
                    self.assertNotIn(
                        pattern.lower(),
                        content,
                        f"Forbidden synthetic rater pattern '{pattern}' found in {pyfile}!"
                    )

    def test_honest_inter_rater_reporting(self):
        """Verify evaluate_inter_rater.py transparently declares single-annotator setup without fake ratings."""
        script_path = PROJECT_ROOT / "scripts" / "evaluate_inter_rater.py"
        self.assertTrue(script_path.exists())
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("human_single_annotator", code)
        self.assertIn("second_human_ratings.csv", code)
        # Ensure it does not fabricate a second rater
        self.assertNotIn("random.choice", code)
        self.assertNotIn("rater_2 =", code)

    def test_compute_dimension_metrics_exactness(self):
        """Test statistical accuracy of compute_dimension_metrics."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        metrics = compute_dimension_metrics(y_true, y_pred)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["exact_agreement"], 1.0)
        self.assertEqual(metrics["within_one"], 1.0)
        self.assertEqual(metrics["spearman"], 1.0)
        self.assertEqual(metrics["pearson"], 1.0)

        # Off by 1 test
        y_pred_off = [2.0, 3.0, 4.0, 5.0, 5.0]
        metrics_off = compute_dimension_metrics(y_true, y_pred_off)
        self.assertAlmostEqual(metrics_off["mae"], 0.8, places=2)
        self.assertAlmostEqual(metrics_off["within_one"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
