# Baseline Architecture Record (Phase 0 Freeze)

## Sequential Pipeline Architecture

```text
Customer message
      ↓
Intent classifier (8-class: device_issue, software_bug, account_security, connectivity, billing_purchase, product_inquiry, general_feedback, other)
      ↓
Retriever (TF-IDF vectorizer, sublinear TF, cosine similarity over historical pairs)
      ↓
Escalation engine (Deterministic safety rules + intent confidence + fatigue guards)
      ↓
LLM reply generator (Prompt template with retrieved historical evidence and persona)
      ↓
Judge / evaluator (Rubric-based evaluation across groundedness, helpfulness, relevance, brand_alignment, safety)
```

## Freezing Mandate
As established in Phase 0:
Do not optimize the model until the evaluation pipeline and dataset splitting are verified, immutable, and trustworthy.
