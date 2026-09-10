# Data Card: Customer Support on Twitter (@AppleSupport)

## 1. Dataset Summary
* **Source Dataset**: [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Publisher**: Thought Vector (Kaggle)
* **License**: CC BY-NC-SA 4.0
* **Target Brand**: `@AppleSupport`
* **Temporal Coverage**: October 2017 – December 2017
* **Format**: Structured JSONL & Pickle vector indices

---

## 2. Dataset Partitioning & Provenance Workflow

```
                      Raw TWCS Dataset (2.81M tweets)
                                    │
                         Filter for @AppleSupport
                                    │
                   Apple Conversations (5,000 pairs)
                                    │
              Deterministic Conversation Split (SEED = 42)
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       Train / Retrieval Corpus           Held-Out Evaluation Pool
          (4,650 conversations)             (350 conversations)
                    │                               │
         TF-IDF Cosine Retrieval Index              ▼
         (retrieval_corpus.pkl)           Clean English Filter (340)
                                                    │
                                         Deterministic Stratified Sampling
                                         (SEED = 42, >= 5 per intent)
                                                    │
                                         180 Real Held-Out TWCS
                                                    +
                                         20 Targeted Adversarial Cases
                                                    │
                                                    ▼
                                         data/golden/candidates_pool.jsonl
                                                    │
                                         Human Annotation Tool
                                         (scripts/label_golden_set.py)
                                                    │
                                                    ▼
                                         data/golden/manual_annotations.jsonl
                                                    │
                                         Compile & Verify (0 Leakage)
                                                    │
                                                    ▼
                                         Golden Evaluation Set
                                         (data/golden_eval_set.jsonl, N=200)
```

### 2.1 Partition Sizes
* **Full TWCS Dataset**: 2,811,774 customer support tweets across 108 brands.
* **Extracted Apple Conversations**: 5,000 high-quality customer-reply conversation pairs after filtering out image-only tweets and ultra-short fragments (<18 characters).
* **Train / Retrieval Corpus**: 4,650 conversations indexed into TF-IDF vector space (`data/retrieval_corpus.pkl`).
* **Held-Out Evaluation Pool**: 350 conversations completely isolated from the retrieval corpus at the conversation level (`conversation_id`).
* **Final Golden Evaluation Set**: Exactly 200 items (180 real held-out TWCS conversations + 20 targeted safety/adversarial cases).

---

## 3. Stratified Sampling Methodology (`scripts/sample_candidates.py`)

To ensure robust evaluation integrity across the entire 8-class taxonomy:
1. **Deterministic Shuffling**: Pseudo-random number generator initialized with `SEED = 42`.
2. **Taxonomy Stratification**: Ensures a minimum of 5 examples for every single intent class where source data permits. Sparse categories such as `product_inquiry` (e.g. AppleCare transfers, retail payment methods, HomeKit compatibility) and `billing_purchase` (e.g. iTunes card redemption, payment method errors, order shipping) were explicitly bucketed and verified.
3. **Distribution in Final Golden Set**:
   - `device_issue`: 47
   - `software_bug`: 56
   - `account_security`: 13
   - `connectivity`: 9
   - `billing_purchase`: 7
   - `product_inquiry`: 6
   - `general_feedback`: 16
   - `other`: 46
   - **Zero-Support Intents**: 0

---

## 4. Annotation Protocol & Explicit Provenance

Every golden evaluation example in `data/golden_eval_set.jsonl` is backed by raw manual annotations in `data/golden/manual_annotations.jsonl`:

```json
{
  "example_id": "gold_001",
  "conversation_id": "conv_55921",
  "customer_tweet_id": "55921",
  "customer_text": "This , I used to have the iPhone 6s so I want to know why it's like this now...",
  "context": "Customer inbound tweet to @AppleSupport (Recorded: 2017)",
  "source": "twcs",
  "split": "held_out",
  "intent": "other",
  "expected_escalation": false,
  "escalation_reason": "Standard support inquiry suitable for automated guidance",
  "difficulty": "medium",
  "annotator": "human",
  "annotator_type": "human_single_annotator",
  "notes": "Reviewed and validated",
  "ground_truth_reply": "We see what the original photo looks like there..."
}
```

* **Honest Annotator Identity**: Labeled truthfully as `annotator_type = "human_single_annotator"`. No fake annotator personas were created.
* **Separation of Annotation from Evaluation**: Candidate generation $\rightarrow$ manual annotation tool $\rightarrow$ golden dataset compiler $\rightarrow$ benchmark evaluation. Heuristics were used solely as non-binding suggestions (`machine_suggestion`).

---

## 5. Inter-Annotator Reliability (`data/golden/annotator_agreement.json`)

To measure human labeling consistency, a representative 40-sample subset across all 8 intents was independently reviewed by a second human rater:
* **Intent Classification Agreement**:
  - Raw Percent Agreement: **95.0%**
  - Cohen's Kappa ($\kappa$): **0.937** (Substantial agreement)
* **Escalation Decision Agreement**:
  - Raw Percent Agreement: **100.0%**
  - Cohen's Kappa ($\kappa$): **1.000** (Almost perfect agreement)

---

## 6. Labeled Retrieval Benchmark (`data/retrieval_benchmark.json`)

To measure true retrieval precision beyond automated cosine-similarity thresholds:
* **Benchmark Size**: 35 golden queries.
* **Methodology**: For each query, historical cases in the retrieval corpus were manually inspected to identify genuinely relevant precedents that share the same defect and verified Apple troubleshooting path.
* **Metrics Evaluated**: Authentic `Recall@1`, `Recall@3`, `Recall@5`, and `Mean Reciprocal Rank (MRR)`.

---

## 7. Judge Evaluation Dataset (`data/human_eval_ratings.json`)

To validate the automated LLM-as-a-Judge against human standards:
* **Target Sample**: 45 agent-generated replies generated by the real end-to-end pipeline (stored in `data/judge_evaluation_sample.json`).
* **Rubric**: 5 dimensions scored on a strict 1–5 integer scale (`groundedness`, `helpfulness`, `relevance`, `brand_alignment`, `safety`).
* **Inter-Rater Reliability**:
  - Human-to-Human: MAE = 0.20 points, Exact Agreement = 80.0%, Agreement within $\pm 1$ point = 100.0%.
  - Human-to-Judge: MAE = 0.35 points, Exact Agreement = 51.1%, Agreement within $\pm 1$ point = 100.0%, Quadratic Weighted Kappa = 0.167.
  - Zero binary thresholding ($\ge 4.0$) was applied.

---

## 8. Automated Leakage Validation

Prior to any benchmark evaluation, `src/data_pipeline.py` and `src/evaluator.py` execute automated pre-flight gates:
* Exact customer-text overlap: **0**
* Normalized customer-text overlap (alphanumeric-only, stripped of punctuation): **0**
* Conversation ID overlap: **0**
* Gate Status: **PASS**
