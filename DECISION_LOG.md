# Architectural Decision Log: 13 Non-Obvious Engineering Decisions

This document records the key architectural, methodological, and product trade-offs made during the development and evaluation of the `@AppleSupport` AI Support Agent.

---

### 1. Brand Selection: Targeting @AppleSupport over @AmazonHelp or @SpotifyCares
* **Decision**: Selected `@AppleSupport` as the single brand focus for the support agent.
* **Why**: Apple Support combines high conversational volume (>100k conversations in TWCS), diverse multi-tier technical troubleshooting across hardware and OS layers, distinctive brand tone guidelines, and high-stakes privacy/safety edge cases (thermal incidents, locked Apple IDs).
* **Alternative Considered**: `@AmazonHelp` (delivery logistics) or `@SpotifyCares` (streaming service).
* **Why Rejected**: Amazon inquiries are dominated by private package tracking and delivery logistics requiring internal database lookups rather than public conversational guidance. Spotify has a much narrower technical scope. Apple provides the ideal proving ground for intent classification, safety boundaries, and technical grounding.

---

### 2. Taxonomy Scope: Domain-Tailored 8-Class Intent Taxonomy vs. Banking77 (77 Intents) vs. 3-Class Coarse Set
* **Decision**: Designed an 8-class domain-tailored intent taxonomy (`device_issue`, `software_bug`, `account_security`, `connectivity`, `billing_purchase`, `product_inquiry`, `general_feedback`, `other`).
* **Why**: 8 intents cleanly partition consumer electronics support into actionable business workflows with distinct troubleshooting procedures and escalation protocols.
* **Alternative Considered**: Adopting the Banking77 benchmark (77 intents) or a generic 3-class bucket (`issue`, `inquiry`, `feedback`).
* **Why Rejected**: Banking77 has extreme label sparsity and semantic overlap that does not fit consumer hardware support. A 3-class system is too coarse to determine whether an issue requires hardware repair guidance or account recovery portals.

---

### 3. Data Splitting: Conversation-Level Splitting vs. Tweet-Level Random Splitting
* **Decision**: Enforced strict conversation-level splitting (`conversation_id` / `customer_tweet_id`) with a fixed seed (`SEED = 42`), separating 4,650 retrieval pairs from 350 held-out evaluation pairs.
* **Why**: Prevents conversational leakage where an agent sees the customer's opening tweet in training and the follow-up tweet in evaluation, which artificially inflates retrieval and generation metrics.
* **Alternative Considered**: Uniform random row-level splitting of the dataset.
* **Why Rejected**: Random splitting scatters related tweets from the same thread across train and test sets, violating independence assumptions and masking evaluation leakage.

---

### 4. Golden Set Construction: Real Human Ground Truth vs. Heuristic Rule Pseudo-Labels
* **Decision**: Built the 200-sample Golden Evaluation Set via deterministic stratified sampling from the held-out pool, labeled by a human engineer using a dedicated CLI tool (`scripts/label_golden_set.py`), storing raw annotations separately in `data/golden/manual_annotations.jsonl`.
* **Why**: Heuristic labels suffer from systematic rule biases and zero coverage on sparse classes (e.g., 0 `product_inquiry` samples under naive regexes). Real human ground truth is indispensable for evaluation integrity.
* **Alternative Considered**: Generating ground truth using deterministic regexes (`classify_text_heuristically`).
* **Why Rejected**: Presenting heuristic guesses as golden ground truth invalidates the entire benchmark.

---

### 5. Automated Leakage Detection: Hard Pre-Flight Gate vs. Informal Manual Inspection
* **Decision**: Implemented an automated pre-flight self-validation gate checking exact text matches, normalized text matches (casing and punctuation stripped), and conversation ID overlap before evaluation runs (`scripts/check_leakage.py`).
* **Why**: Guarantees that no evaluation query exists in the retrieval index, ensuring scientific credibility and defense under reviewer audit.
* **Alternative Considered**: Relying on manual assurance that splits were performed correctly.
* **Why Rejected**: Unchecked data processing often introduces subtle overlap that compromises evaluation validity.

---

### 6. Retrieval Architecture: TF-IDF with Evidence ID Tracking vs. Dense Neural Index
* **Decision**: Used TF-IDF vectorization with sublinear term frequency and cosine similarity, returning structured `evidence_ids` and automatic index rebuild if the index file is missing.
* **Why**: Executes in milliseconds locally, requires zero external GPU infrastructure or heavy embedding dependencies, and provides exact lexical matching for specific error codes and device models (e.g., "iOS 11.1.2", "iPhone 7 Plus").
* **Alternative Considered**: Dense neural retrieval (Sentence-Transformers / FAISS).
* **Why Rejected**: Adding large embedding models increases repository clone size, introduces heavy compiler/torch dependencies on clean environments, and slows pipeline reproduction beyond the 15-minute budget.

---

### 7. Retrieval Evaluation: Separating Heuristic Hits from Labeled Benchmarks
* **Decision**: Renamed heuristic cosine-overlap checks to `Heuristic Retrieval Hit@K`, and created a dedicated 35-sample human-judged retrieval benchmark (`data/retrieval_benchmark.json`) to report authentic `Recall@1`, `Recall@3`, `Recall@5`, and `MRR`.
* **Why**: Calling an automated cosine similarity $\ge 0.15$ threshold "Recall@K" is scientifically misleading.
* **Alternative Considered**: Reporting heuristic hits as standard Recall@K.
* **Why Rejected**: Reviewers will rightfully challenge self-defined heuristic recall metrics.

---

### 8. Pipeline Evaluation Isolation: Strict Separation of Component Oracle vs. End-to-End Runtime
* **Decision**: Built two independent evaluation pathways: component-level oracle (gold labels) and end-to-end (pipeline predictions driving downstream retrieval, escalation, and reply generation with `gold_intent_injected = False`).
* **Why**: In production, the agent never receives the gold intent. Injecting gold intent into downstream stages masks cascading classification errors.
* **Alternative Considered**: Feeding gold intent to the agent during full benchmark runs.
* **Why Rejected**: Conceals pipeline error propagation and produces artificially inflated headline scores.

---

### 9. Baseline Hierarchy: Deterministic Majority Floor & Classical ML vs. Stochastic Random Choice
* **Decision**: Implemented deterministic Majority Class (`software_bug`) as Baseline 1 (Macro-F1: 0.055), and TF-IDF + Logistic Regression (`SEED = 42`) as Baseline 2 (Macro-F1: 0.251, Accuracy: 29.0%).
* **Why**: Provides a rigorous classical ML floor that any advanced model must prove it surpasses.
* **Alternative Considered**: Uniform random intent sampling (`random.choice(INTENT_NAMES)`).
* **Why Rejected**: Random choice baseline is stochastic, non-informative, and easily beaten. A deterministic majority baseline is the standard statistical benchmark.

---

### 10. Escalation Architecture: Deterministic Safety Guardrails First, Probabilistic Reasoning Second
* **Decision**: Physical safety hazards (swelling, fire, smoke, sparks), legal threats, unauthorized financial disputes, and explicit human requests bypass probabilistic models and escalate deterministically via semantic regexes.
* **Why**: Life-safety and regulatory hazards must have zero false negatives. Subjecting a smoking charger or expanding battery to probabilistic LLM temperature sampling introduces unacceptable safety risks.
* **Alternative Considered**: Prompting the LLM to classify safety and decide escalation purely via prompting.
* **Why Rejected**: LLMs are vulnerable to prompt injection, sycophancy, and nondeterministic misses on safety-critical edge cases.

---

### 11. Metric Denominator Rigor: Class-Specific Error Rates vs. Total Dataset Denominators
* **Decision**: Calculated False Escalation Rate as $FP / \text{Actual Auto-Handle}$ ($42/166 = 25.3\%$) and Missed Escalation Rate as $FN / \text{Actual Escalate}$ ($12/34 = 35.3\%$), explicitly printing numerators and denominators.
* **Why**: Dividing false escalations by the total evaluation set ($42/200 = 21.0\%$) artificially deflates the error rate and misrepresents operational performance.
* **Alternative Considered**: Using total dataset size ($N=200$) as the universal denominator.
* **Why Rejected**: Mathematically invalid and misleading to helpdesk operations teams calculating true advisor workload.

---

### 12. Auxiliary Judge Calibration on Actual Agent Replies vs. Synthetic Benchmarks or Binary Thresholding
* **Decision**: Calibrated the LLM judge against 45 actual agent-generated replies scored by a human annotator on a 1–5 scale, computing continuous MAE (1.09) and Quadratic Weighted Kappa (-0.067) rather than applying artificial binary thresholding.
* **Why**: Evaluating the judge on 2017 historical Twitter tweets does not validate whether the judge can detect errors in modern agent-generated responses.
* **Alternative Considered**: Binning scores into $\ge 4.0$ to artificially claim high agreement.
* **Why Rejected**: Artificial thresholding conceals ordinal rating divergence and misleads technical reviewers. The resulting weak correlation ($\rho = -0.147$) is honestly reported as an experimental finding.

---

### 13. Immutable Split Persistence vs. Dynamic Runtime Shuffling
* **Decision**: The data pipeline checks for persisted `data/retrieval_corpus.jsonl` and `data/split_manifest.json` upon startup and refuses to dynamically re-partition the dataset.
* **Why**: Dynamic re-splitting across runs risks subtle shuffle divergence, invalidating golden evaluation sets and introducing invisible training-evaluation leakage.
* **Alternative Considered**: Re-running split logic dynamically every time the data pipeline is instantiated.
* **Why Rejected**: Violates benchmark immutability and reproducibility guarantees.
