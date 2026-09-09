# Architectural Decision Log: 14 Non-Obvious Engineering Decisions

This document records the key architectural, methodological, and product trade-offs made during the development and evaluation of the `@AppleSupport` AI Support Agent.

---

### 1. Brand Selection: Targeting @AppleSupport over @AmazonHelp or @SpotifyCares
* **Decision**: Selected `@AppleSupport` as the single brand focus for the system.
* **Why**: Apple Support combines high customer volume (>100k conversations), diverse technical troubleshooting across hardware and OS layers, distinctive brand tone guidelines, and high-stakes privacy/safety edge cases.
* **Alternative Considered**: `@AmazonHelp` (delivery logistics) or `@SpotifyCares` (streaming service).
* **Why Rejected**: Amazon inquiries are dominated by private package tracking and delivery driver logistics that require proprietary database lookups rather than public conversational support. Spotify has a narrower technical scope. Apple represents the ideal proving ground for intent classification, safety boundaries, and technical grounding.

---

### 2. Taxonomy Scope: Custom 8-Class Intent Taxonomy vs. Banking77 (77 Intents) vs. 3-Class Coarse Set
* **Decision**: Designed an 8-class domain-tailored intent taxonomy (`device_issue`, `software_bug`, `account_security`, `connectivity`, `billing_purchase`, `product_inquiry`, `general_feedback`, `other`).
* **Why**: 8 intents cleanly partition consumer electronics support into actionable business workflows with distinct troubleshooting procedures and escalation protocols.
* **Alternative Considered**: Adopting the Banking77 benchmark (77 intents) or a generic 3-class bucket (`issue`, `inquiry`, `feedback`).
* **Why Rejected**: Banking77 has extreme label sparsity and semantic overlap that does not fit consumer hardware support. A 3-class system is too coarse to determine whether an issue requires hardware repair guidance or account recovery portals.

---

### 3. Data Splitting: Conversation-Level Splitting vs. Tweet-Level Random Splitting
* **Decision**: Enforced strict conversation-level splitting (`conversation_id` / `customer_tweet_id`) with a fixed seed (`SEED = 42`), separating 4,650 retrieval pairs from 350 held-out evaluation pairs.
* **Why**: Prevents conversational leakage where an agent sees the customer's opening tweet in training and the follow-up tweet in evaluation, which inflates retrieval metrics artificially.
* **Alternative Considered**: Uniform random row-level splitting of the dataset.
* **Why Rejected**: Random splitting scatters related tweets from the same thread across train and test sets, violating independence assumptions and masking evaluation leakage.

---

### 4. Golden Set Provenance: Held-Out Empirical Conversations vs. Synthetic Intent Dictionary
* **Decision**: Built the 200-sample Golden Evaluation Set from 180 real held-out AppleSupport customer tweets plus 20 targeted adversarial cases.
* **Why**: Handcrafted synthetic dictionaries generate artificially clean, grammatically uniform sentences that fail to test how the agent handles customer typos, colloquial slang, and emotional venting.
* **Alternative Considered**: Hardcoding 25 synthetic sentences per intent from a dictionary.
* **Why Rejected**: Synthetic customer messages create an artificial benchmark detached from real-world Twitter distribution shift.

---

### 5. Automated Leakage Detection: Hard Gate vs. Informal Manual Inspection
* **Decision**: Implemented an automated leakage gate (`check_evaluation_leakage`) checking exact text matches, normalized text matches (casing and punctuation stripped), and conversation ID overlap before evaluation runs.
* **Why**: Guarantees that no evaluation query exists in the retrieval index, ensuring scientific credibility and defense under interview scrutiny.
* **Alternative Considered**: Relying on manual assurance that splits were performed correctly.
* **Why Rejected**: Unchecked data processing often introduces subtle overlap (e.g. duplicate customer inquiries across threads) that compromises evaluation validity.

---

### 6. Retrieval Architecture: TF-IDF with Evidence ID Tracking vs. Dense Embeddings
* **Decision**: Used TF-IDF vectorization with sublinear term frequency and cosine similarity, returning structured `evidence_ids`.
* **Why**: Runs in milliseconds locally, requires zero external GPU infrastructure or heavy embedding dependencies, and provides exact lexical matching for specific error codes and device models (e.g., "iOS 11.1.2", "iPhone 7 Plus").
* **Alternative Considered**: Dense neural retrieval (Sentence-Transformers / FAISS).
* **Why Rejected**: Adding large embedding models increases repository clone size, introduces torch/C++ compiler dependencies on Windows, and slows pipeline reproduction beyond the 15-minute budget.

---

### 7. Evaluation Protocol: Strict Separation of Component-Level vs. End-to-End Evaluation
* **Decision**: Built two independent evaluation pathways: component-level (oracle gold labels) and end-to-end (agent predictions driving downstream retrieval and escalation).
* **Why**: In production, the agent never receives the gold intent. Injecting the gold intent into the generation or escalation stages masks cascading classification errors.
* **Alternative Considered**: Feeding gold intent to the agent during full benchmark runs.
* **Why Rejected**: Conceals pipeline error propagation and produces artificially inflated headline scores.

---

### 8. Baseline Hierarchy: Deterministic Majority Class & Classical ML vs. Random Choice
* **Decision**: Implemented deterministic Majority Class (`software_bug`) as Baseline 1, and TF-IDF + Logistic Regression (`SEED = 42`) as Baseline 2.
* **Why**: Provides a rigorous classical ML floor (65.5% accuracy, 0.618 Macro-F1) that any advanced model must prove it surpasses.
* **Alternative Considered**: Uniform random intent sampling (`random.choice(INTENT_NAMES)`).
* **Why Rejected**: Random choice baseline is stochastic, non-informative, and easily beaten. A deterministic majority baseline is the standard statistical benchmark.

---

### 9. Escalation Architecture: Deterministic Hard Guardrails First, Probabilistic Reasoning Second
* **Decision**: Physical safety hazards (swelling, fire, smoke, sparks), legal threats, and explicit human requests bypass probabilistic models and escalate deterministically via lemma-based regexes.
* **Why**: Life-safety hazards must have zero false negatives. Subjecting a smoking charger or expanding battery to probabilistic LLM temperature sampling introduces unacceptable safety risks.
* **Alternative Considered**: Prompting the LLM to classify safety and decide escalation purely via prompting.
* **Why Rejected**: LLMs are vulnerable to prompt injection, sycophancy, and nondeterministic misses on safety-critical edge cases.

---

### 10. Metric Denominator Integrity: Class-Specific Denominators vs. Total Dataset
* **Decision**: Calculated False Escalation Rate as $FP / \text{Actual Auto-Handle}$ ($34/164 = 20.7\%$) and Unsafe Auto-Handle Rate as $FN / \text{Actual Escalate}$ ($6/36 = 16.7\%$), explicitly printing numerators and denominators.
* **Why**: Dividing false escalations by the total evaluation set ($34/200 = 17.0\%$) artificially deflates the error rate and misrepresents operational performance.
* **Alternative Considered**: Using total dataset size ($N=200$) as the universal denominator.
* **Why Rejected**: Mathematically invalid and misleading to helpdesk managers calculating true analyst workload.

---

### 11. Judge Design: 5-Dimension Structured Rubric without Mandatory Chain-of-Thought
* **Decision**: The automated judge scores 5 explicit dimensions (`groundedness`, `helpfulness`, `relevance`, `brand_alignment`, `safety`) on a 1–5 scale with a concise 1–2 sentence rationale.
* **Why**: Evaluates multi-dimensional support quality without requiring verbose reasoning chains that increase API token latency and latency costs.
* **Alternative Considered**: Single overall binary score ("good" vs "bad") or mandatory multi-paragraph Chain-of-Thought.
* **Why Rejected**: Binary scoring misses subtle tone or groundedness flaws; long CoT increases latency without improving inter-rater correlation.

---

### 12. Human Judge Validation: Multi-Rater Human Dataset ($N=50$) vs. Simulated Scores
* **Decision**: Collected authentic human annotations from two independent raters across 50 examples, calculating real Pearson $r$, MAE, and Cohen's Kappa.
* **Why**: Automated judges cannot be trusted without empirical validation against genuine human ratings.
* **Alternative Considered**: Generating synthetic human ratings via `random.choice([4, 5, 5, 5])`.
* **Why Rejected**: Fabricating simulated human scores destroys evaluation integrity and cannot be defended in an interview.

---

### 13. Offline Strategy: Honest Local Fallback vs. Masked Baseline Substitution
* **Decision**: In offline mode, the pipeline executes deterministic local components (Logistic Regression, TF-IDF retrieval, template generation, heuristic judge) and labels outputs explicitly as `[Offline Local / Baseline]`.
* **Why**: Reviewers often run code without API keys. The system must run flawlessly without crashing, but must NEVER deceive the user by labeling a heuristic baseline as the "Primary Agent".
* **Alternative Considered**: Silently substituting simple keyword baselines into the "Primary Agent" output slot.
* **Why Rejected**: Dishonest engineering that misrepresents system capabilities.

---

### 14. Lexical Metrics (BLEU/ROUGE): Descriptive Context, Not Primary Optimization Goal
* **Decision**: Reported BLEU-1 and ROUGE-L for academic completeness, but based quality assessments on grounded rubric scores.
* **Why**: In customer support, two responses with identical operational meaning (e.g. *"Head to Settings > General > Update"* vs *"Please check your software update settings"*) have near-zero n-gram overlap.
* **Alternative Considered**: Optimizing generator prompts to maximize BLEU score against historical reference tweets.
* **Why Rejected**: Produces rigid, over-fitted responses that mimic 2017 phrasing rather than prioritizing helpfulness and safety.
