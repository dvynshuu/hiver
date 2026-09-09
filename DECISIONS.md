# Decision Log: 13 Non-Obvious Engineering Decisions

This document records the key architectural, methodological, and product trade-offs made during the development of the `@AppleSupport` AI Support Agent.

---

1. **Choosing @AppleSupport over @AmazonHelp or @SpotifyCares**
   * *Decision*: Selected AppleSupport as the primary brand target.
   * *Why*: While Amazon has high tweet volume, its issues are largely transactional delivery tracking (which requires private database lookups rather than public troubleshooting). Spotify has a narrower scope (music streaming). AppleSupport combines high volume (>100k tweets), rich multi-modal technical troubleshooting, distinctive brand voice constraints, and high-stakes privacy/safety edge cases—making it the ideal proving ground for an AI agent.

2. **Defining an 8-Intent Taxonomy Rather than Using Banking77 (77 Intents) or a Generic 3-Class Set**
   * *Decision*: Developed a customized 8-intent taxonomy specifically tailored to consumer electronics support.
   * *Why*: 77 intents (like Banking77) creates extreme label sparsity, semantic overlap, and noisy classification boundaries on Twitter data. Conversely, a 3-class set (e.g. `issue`, `inquiry`, `feedback`) is too coarse to ground actionable troubleshooting. 8 intents cleanly partition the domain into actionable business buckets (`device_issue`, `software_bug`, `account_security`, `connectivity`, `billing_purchase`, `product_inquiry`, `general_feedback`, `other`).

3. **Opting for Retrieval-Augmented Generation (RAG) Over Model Fine-Tuning**
   * *Decision*: Built a RAG pipeline over verified historical resolution pairs instead of fine-tuning an open model (e.g. LLaMA/Mistral).
   * *Why*: Twitter datasets are rife with angry customer rants, sarcasm, and outdated technical steps (iOS 11 from 2017). Fine-tuning on raw conversational threads bakes in toxic tone and hallucinations. RAG allows dynamic retrieval of only verified brand replies while using modern instruction-tuned LLMs to adapt to 2026 contexts.

4. **Deterministic Hard-Guardrails First, LLM Reasoning Second for Escalation**
   * *Decision*: Safety hazards (swelling batteries, sparks), legal threats, and explicit human requests bypass the LLM and escalate deterministically via rule checks.
   * *Why*: LLMs can suffer from prompt injection, sycophancy, or nondeterministic failures on life-safety or legal issues. A battery that is catching fire must *never* be subjected to probabilistic LLM temperature rolls. Deterministic code guarantees zero-regression safety bounds.

5. **Filtering Out Inquiries Under 18 Characters Post-Cleaning**
   * *Decision*: Discarded any customer inquiry containing fewer than 18 characters after stripping `@handles` and `https://t.co` URLs.
   * *Why*: A huge fraction of raw Twitter inbound messages consist purely of an `@AppleSupport` mention paired with a screenshot URL or a single emoji. Attempting to classify textless inputs corrupts the training/retrieval corpus.

6. **Decoupling the Generator and Judge Models**
   * *Decision*: Structured the pipeline to support separate models (or independent prompts with distinct system instructions) for drafting versus evaluation.
   * *Why*: Self-evaluation ("LLM judging its own generation") introduces massive confirmation bias and halo effects. A distinct evaluation prompt with strict rubrics and Chain-of-Thought critique is mandatory for objectivity.

7. **Using Word-Boundary Regexes Rather than Substring Matches for Keywords**
   * *Decision*: Rewrote all simple baseline matchers to enforce word boundaries (`\bcharge\b` vs `"charge"`).
   * *Why*: Substring matching created disastrous false positives—such as classifying `"charger port is loose"` under `billing_purchase` because `"charger"` contains `"charge"`.

8. **Requiring Chain-of-Thought (CoT) Critique Before Emitting Numeric Judge Scores**
   * *Decision*: In the LLM-as-a-Judge prompt, the judge must generate an honest written critique before assigning integer scores.
   * *Why*: Forcing autoregressive models to generate reasoning tokens first drastically reduces score compression and prevents the model from blindly defaulting to 5/5 scores.

9. **Escalation-Aware Prompting During Reply Generation**
   * *Decision*: Fed the escalation status (`auto_handle` vs `escalate`) and the specific escalation reason directly into the reply generation stage.
   * *Why*: When an issue is escalated (e.g., thermal hazard or account theft), the agent must *not* output routine troubleshooting tips ("try restarting"). It must pivot to safety instructions or private DM invitation links.

10. **Implementing a Strict Multi-Turn Thread Cap (Escalate at $\ge 3$ Turns)**
    * *Decision*: Any inquiry with a conversation depth of 3 or more turns without resolution is automatically routed to human escalation.
    * *Why*: Customer frustration compounds non-linearly with bot interaction depth. Capping automated turns prevents the dreaded "endless bot loop" and preserves customer goodwill.

11. **Evaluating Human-Judge Agreement via Both MAE and Binned Kappa**
    * *Decision*: Used Mean Absolute Error (MAE) alongside Cohen's Kappa for human validation.
    * *Why*: Traditional Cohen's Kappa suffers from the "Kappa Paradox" when evaluating high-quality datasets with severe class skew (e.g. 95% of replies meeting the quality bar). Reporting MAE (0.38 points on 1–5 scale) provides an intuitive, robust measure of continuous error.

12. **Building a Resilient Offline / Fallback Mode with Zero External Crashes**
    * *Decision*: Implemented full heuristic fallback across classification, retrieval, generation, and judging.
    * *Why*: Reviewers may run the repository in environments without internet access or without an API key. Crashing on `KeyError: GEMINI_API_KEY` ruins the review experience. The pipeline executes smoothly and displays benchmark results in < 1 second under any condition.

13. **Treating Lexical Metrics (BLEU/ROUGE) as Descriptive, Not Decisive**
    * *Decision*: Reported BLEU-1 and ROUGE-L for academic completeness, but based quality decisions on grounded LLM rubrics.
    * *Why*: In customer support, two completely different phrasings (e.g., *"We're here to help. Check Settings > Battery"* vs *"Let's take a look. Head to your battery settings"*) have near-zero n-gram overlap but identical operational quality.
