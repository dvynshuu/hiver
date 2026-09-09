# Data Card: Customer Support on Twitter (@AppleSupport)

## 1. Dataset Summary
* **Source Dataset**: [Customer Support on Twitter (TWCS)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Publisher**: Thought Vector (Kaggle)
* **License**: CC BY-NC-SA 4.0
* **Target Brand**: `@AppleSupport`
* **Temporal Coverage**: October 2017 – December 2017
* **Format**: Structured JSONL & Pickle vector indices

---

## 2. Dataset Partitioning & Provenance

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
                                         Sample 180 Real TWCS
                                                    +
                                         20 Adversarial Cases
                                                    │
                                                    ▼
                                          Golden Evaluation Set
                                        (200 examples, leak-free)
```

### 2.1 Original vs. Processed Sizes
* **Full TWCS Dataset**: 2,811,774 customer support tweets across 108 brands.
* **Extracted Apple Conversations**: 5,000 high-quality customer-reply conversation pairs after filtering out image-only tweets and ultra-short fragments (<18 characters).
* **Train / Retrieval Corpus**: 4,650 conversations indexed into TF-IDF vector space for grounded reply generation.
* **Held-Out Evaluation Pool**: 350 conversations completely isolated from the retrieval corpus.
* **Final Golden Evaluation Set**: 200 items (180 real held-out TWCS conversations + 20 targeted safety/adversarial cases).

---

## 3. Train / Retrieval vs. Evaluation Split Methodology

To strictly prevent data leakage and benchmark gaming:
1. **Conversation-Level Splitting**: Splitting was performed at the **conversation level** (`conversation_id` / `customer_tweet_id`), ensuring that customer tweets and brand replies from the same thread never cross the boundary.
2. **Fixed Random Seed**: Splitting uses deterministic pseudo-random shuffling (`SEED = 42`).
3. **Retrieval Index Isolation**: The TF-IDF vectorizer and cosine similarity corpus are fitted **exclusively on the 4,650 retrieval pairs**. The evaluation set is never observed during index fitting or retrieval fitting.

---

## 4. Automated Leakage Validation

Prior to any benchmark evaluation, `src/data_pipeline.py` executes `check_evaluation_leakage()`:
* **Exact customer-text overlap**: 0 matches
* **Normalized customer-text overlap** (lowercased, alphanumeric-only, stripped of punctuation): 0 matches
* **Conversation ID overlap**: 0 matches
* **Customer Tweet ID overlap**: 0 matches
* **Verification Status**: `PASS`

---

## 5. Golden Set Schema & Metadata

Every golden evaluation example in `data/golden_eval_set.jsonl` retains full provenance:

```json
{
  "example_id": "gold_001",
  "conversation_id": "conv_117936",
  "customer_tweet_id": "117936",
  "customer_text": "Hey I'm having trouble with a defective iPhone...",
  "context": "Customer inbound tweet on Twitter/X to @AppleSupport",
  "ground_truth_intent": "device_issue",
  "expected_escalation": false,
  "ground_truth_escalation": "auto_handle",
  "escalation_trigger": "standard_support",
  "escalation_reason": "Standard support inquiry suitable for automated guidance",
  "difficulty": "medium",
  "ground_truth_reply": "We can certainly take a look at this with you...",
  "source": "twcs",
  "split": "golden_eval"
}
```

---

## 6. Human Evaluation Dataset (`data/human_eval_ratings.json`)

To validate the LLM-as-a-Judge against human judgment:
* **Sample Size**: 50 examples from the Golden Evaluation Set.
* **Annotators**: Two independent raters (`rater_1`, `rater_2`).
* **Rubric**: 5 dimensions on a 1–5 scale (`groundedness`, `helpfulness`, `relevance`, `brand_alignment`, `safety`).
* **Inter-Rater Statistics**: Human-to-human agreement and human-to-LLM agreement are reported via Pearson correlation ($r$), Spearman rank correlation ($\rho$), Mean Absolute Error (MAE), and Cohen's Kappa ($\kappa$).
* **Provenance**: No synthetic or random ratings. All scores reflect authentic evaluation criteria.

---

## 7. Known Limitations & Potential Risks

1. **Temporal Domain Shift**: The TWCS dataset dates from late 2017 (iOS 11 era). Inquiries refer to iPhone 7/8, Touch ID, and iTunes. Modern support refers to iOS 17/18, Face ID, and AppleCare+ web portals.
2. **Missing Modalities**: ~30% of original Twitter inquiries included screenshots or photos of physical damage. TWCS anonymized or stripped URLs to media attachments, limiting customer text to textual descriptions.
3. **Class Skew**: In live Twitter operations, hardware glitches and iOS bugs dominate inbound volume (~60–70%), while account security incidents are rarer (~5%). The evaluation set mirrors real distribution rather than artificial uniform balance.
4. **Anonymization Artifacts**: Twitter handles were anonymized to numerical IDs (e.g. `@115854`), which were cleaned during preprocessing.
