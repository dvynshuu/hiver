# Golden Evaluation Set Labelling Guide

This guide details the methodology, intent definitions, escalation criteria, and annotation protocols used to create the 200-sample Golden Evaluation Set for the `@AppleSupport` AI agent.

---

## 1. Objective & Scope

The goal of this evaluation set is to provide a reliable, leak-free benchmark for:
1. **Intent Classification**: 8-class classification (`device_issue`, `software_bug`, `account_security`, `connectivity`, `billing_purchase`, `product_inquiry`, `general_feedback`, `other`).
2. **Reply Quality**: Evaluating generated replies against real customer support standards on a 5-dimension rubric (Relevance, Helpfulness, Tone, Groundedness, Completeness).
3. **Escalation Decision**: Testing whether the agent correctly auto-handles resolvable issues and escalates sensitive, high-risk, or high-frustration issues.
4. **LLM-as-Judge Validation**: Ground truth human ratings across 50 calibration examples to measure human-judge inter-rater agreement (Cohen's Kappa and Pearson correlation).

---

## 2. Sampling Strategy

To prevent skew and ensure rigorous testing across the problem space, we used **Stratified Sampling**:
- **Dataset Source**: `thoughtvector/customer-support-on-twitter` (AppleSupport conversations).
- **Stratification**: 25 examples per intent class ($8 \times 25 = 200$ examples total).
- **Difficulty Partition**:
  - **Easy (40%)**: Explicit keywords, single clear intent, standard troubleshooting scenario.
  - **Medium (40%)**: Conversational phrasing, compound questions, informal slang, subtle symptoms.
  - **Hard (20%)**: Sarcasm, conflicting signals, emotional venting masking a technical issue, edge-case safety/fraud risks.
- **Escalation Split**:
  - **Auto-Handle**: 128 examples (64%)
  - **Escalate**: 72 examples (36%)

---

## 3. Intent Annotation Schema

| Intent Code | Category | Definition | Boundary Rules |
| :--- | :--- | :--- | :--- |
| `device_issue` | Hardware & Physical | Battery degradation, display flicker, cracked glass, speaker distortion, physical buttons, charging ports, overheating. | If hardware is impaired due to a software update (e.g. "update made battery drain"), label `device_issue` if the primary complaint is battery life, or `software_bug` if update installation failed. |
| `software_bug` | OS & Applications | System crashes, boot loops, freeze on Apple logo, app crash, iCloud sync errors, update install errors. | If the issue is exclusively about an App Store purchase/subscription, use `billing_purchase`. |
| `account_security` | Apple ID & Auth | Apple ID locked, 2FA code delivery failures, suspected unauthorized sign-ins, forgotten passcodes. | Always escalated unless it is a general FAQ on how to create an Apple ID. |
| `connectivity` | Wireless & Radio | Wi-Fi disconnects, Bluetooth pairing, AirPods dropout, AirDrop, cellular "No Service", hotspot errors. | If the device won't connect physically via cable, use `device_issue`. |
| `billing_purchase` | Commerce & Subscriptions | Unauthorized credit card charges, refund requests, duplicate charges, in-app purchases, payment methods. | Any dispute involving unauthorized charges is flagged for escalation. |
| `product_inquiry` | Pre-Purchase & Specs | Compatibility (e.g. Apple Pencil models), trade-in values, warranty coverage, AppleCare policy, release dates. | Purely informational inquiries; virtually all should be auto-handled. |
| `general_feedback` | Sentiment & Opinions | Expressing frustration with policies, store staff experiences, praise for helpful staff, general brand sentiment. | If a customer vents angrily without asking for troubleshooting, classify as `general_feedback`. |
| `other` | Conversational / Out-of-Scope | Salutations, spam, single emojis, unintelligible text, off-topic conversational fillers. | Canned polite response or request for clarification. |

---

## 4. Escalation Policy

An example is marked `escalate` if ANY of the following criteria are met:
1. **Safety & Hazard**: Any mention of device smoking, burning, swelling battery, or physical injury.
2. **Legal / Regulatory Risk**: Threats of litigation, lawyer consultation, or formal complaints to regulators.
3. **Explicit Agent Request**: Customer explicitly demands a manager, human representative, or supervisor.
4. **PII / Authentication Boundary**: Issues requiring handling of credentials, bank cards, or 2FA overrides.
5. **High-Value Financial Disputes**: Accusations of fraud, recurring unauthorized charges, or refusal of automatic refund.
6. **Multi-Turn Frustration**: Inquiries showing 3+ failed automated resolution attempts.

All other routine troubleshooting (reboots, network resets, setting adjustments, compatibility checks) are marked `auto_handle`.

---

## 5. Human Calibration for LLM-as-Judge

To assess judge reliability:
- 50 items were rated manually by an expert human annotator across the 5 dimensions on a 1–5 Likert scale.
- The LLM judge (`gemini-2.5-pro` or `gemini-2.0-flash`) is scored against these same 50 cases without seeing the human labels.
- Inter-annotator agreement is computed via **Cohen's Kappa ($\kappa$)** (for binned acceptability) and **Spearman/Pearson correlation** for numeric fidelity.
