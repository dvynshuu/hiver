# Judge Validation Dataset & Provenance Documentation

This directory contains the artifacts for validating the LLM-as-a-Judge against authentic human evaluation.

## Provenance Specification

* **Sample Size**: Exactly 45 examples selected from the 200 held-out golden evaluation set.
* **Deterministic Seed**: `SEED = 42`
* **Source Dataset**: Held-out AppleSupport Twitter (TWCS) conversations with verified zero-leakage isolation from the retrieval corpus.
* **Sampling Methodology**: Stratified deterministic sampling across all 8 taxonomy intents, auto-handled inquiries, escalated inquiries, varying message lengths, and diverse difficulty levels.
* **Agent Reply Generation**: Generated using the actual end-to-end production agent pipeline (`Customer message -> Intent classifier -> Historical retriever -> Escalation decision -> Reply generator`).
* **Zero Gold Injection**: Gold intent labels are **never** passed to the production agent during reply generation; gold labels are preserved strictly for evaluation.
* **Human Rating Method**: Manual human review by an engineering evaluator applying the standardized 1–5 ordinal rubric in `HUMAN_RATING_GUIDE.md`.
* **Rater Count**: 1 human annotator (`rater_type = "human"`). Inter-rater reliability was not measured due to the single-annotator setup; no simulated second annotator was fabricated.
* **LLM Judge Model**: `gemini-3.6-flash` (with deterministic heuristic fallback for offline testing).
* **Generation Date**: 2026-09-10

---

## Artifact Files

1. `judge_validation_sample.jsonl`: The exact 45 golden evaluation examples deterministically selected.
2. `generated_replies.jsonl`: Real agent replies, predicted intents, escalation decisions, and retrieved evidence IDs.
3. `human_review.csv`: Review form exported by `scripts/export_human_review.py` for manual scoring. Ratings are intentionally left blank upon export.
4. `human_ratings.json`: **Input artifact manually produced by a human.** Imported from `human_review.csv` via `scripts/import_human_ratings.py` with strict schema validation. Code NEVER generates, estimates, or infers these ratings.
5. `llm_judge_ratings.json`: Ratings and brief rationales produced independently by the LLM judge.
6. `judge_validation.json`: Dimension-by-dimension and overall comparison metrics (MAE, Spearman rank correlation, Pearson correlation, exact agreement, and within $\pm 1$ agreement).
7. `HUMAN_RATING_GUIDE.md`: Standardized 1–5 ordinal rubric definitions across Groundedness, Helpfulness, Relevance, Brand Alignment, Safety, and Overall quality.

---

## Fail-Closed Operational Policy

If `human_ratings.json` is missing or incomplete, the evaluation pipeline fails closed:
* It does NOT substitute synthetic scores.
* It does NOT use random or heuristic fallbacks.
* It does NOT pretend judge validation succeeded.
* It prints a clear notice:
  ```text
  Human judge validation unavailable:
  human ratings have not been supplied.
  ```
