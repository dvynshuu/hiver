# Golden Evaluation Set Labelling Guide

This guide details the methodology, intent definitions, escalation criteria, and annotation protocols used to create the 200-sample Golden Evaluation Set for the `@AppleSupport` AI agent.

---

## 1. Objective & Scope

The goal of this evaluation set is to provide a reliable, leak-free benchmark for:
1. **Intent Classification**: 8-class classification (`device_issue`, `software_bug`, `account_security`, `connectivity`, `billing_purchase`, `product_inquiry`, `general_feedback`, `other`).
2. **Reply Quality**: Evaluating generated replies against real customer support standards on a 5-dimension rubric (Groundedness, Helpfulness, Relevance, Brand Alignment, Safety).
3. **Escalation Decision**: Testing whether the agent correctly auto-handles resolvable issues and escalates sensitive, high-risk, or high-frustration issues.
4. **LLM-as-Judge Validation**: Ground truth human ratings across 50 calibration examples evaluated by two independent raters (`rater_1`, `rater_2`) to measure human-human and human-judge inter-rater agreement.

---

## 2. Sampling Strategy & Partitioning

To ensure scientific validity and real-world relevance:
- **Dataset Source**: `thoughtvector/customer-support-on-twitter` (AppleSupport conversations).
- **Partitioning**: Conversation-level split (`SEED = 42`) separating 4,650 retrieval conversations from 350 held-out conversations.
- **Composition**:
  - **180 Real Held-Out Conversations**: Drawn from the held-out conversation pool with real context, timestamps, and customer tweet IDs.
  - **20 Targeted Adversarial Cases**: Covering battery swelling variants (swelling, puffy, lifting, smoking), account takeovers (SIM swap, phishing), billing fraud, legal threats, and supervisor demands.
- **Escalation Split**:
  - **Auto-Handle**: 164 examples (82.0%)
  - **Escalate**: 36 examples (18.0%)

---

## 3. Intent Annotation Schema

| Intent Code | Category | Definition | Boundary Rules |
| :--- | :--- | :--- | :--- |
| `device_issue` | Hardware & Physical | Battery degradation, display flicker, cracked glass, speaker distortion, physical buttons, charging ports, overheating, camera. | If hardware is impaired due to a software update (e.g. "update made battery drain"), label `device_issue` if complaint is battery life. If phone is in boot loop or frozen screen, classify as `software_bug`. |
| `software_bug` | OS & Applications | System crashes, boot loops, freeze on Apple logo, app crash, iCloud sync errors, update install errors, storage glitches. | If defect is rooted in software code or iOS update, use `software_bug`. If issue is exclusively about an App Store purchase/subscription, use `billing_purchase`. |
| `account_security` | Apple ID & Auth | Apple ID locked, 2FA code delivery failures, suspected unauthorized sign-ins, forgotten passcodes, SIM swap takeover. | Always escalated or guided to `iforgot.apple.com`. |
| `connectivity` | Wireless & Radio | Wi-Fi disconnects, Bluetooth pairing, AirPods dropout, AirDrop, cellular "No Service", hotspot errors. | Applies to wireless RF protocols. If lightning cable or wired headphones fail physically, use `device_issue`. |
| `billing_purchase` | Commerce & Subscriptions | Unauthorized credit card charges, refund requests, duplicate charges, in-app purchases, payment methods. | Any dispute involving unauthorized charges is flagged for escalation. |
| `product_inquiry` | Pre-Purchase & Specs | Compatibility (e.g. Apple Pencil models), trade-in values, warranty coverage, AppleCare policy, release dates. | Purely informational inquiries; virtually all should be auto-handled. |
| `general_feedback` | Sentiment & Opinions | Expressing frustration with policies, store staff experiences, praise for helpful staff, general brand sentiment. | If a customer vents angrily without asking for troubleshooting, classify as `general_feedback`. |
| `other` | Conversational / Out-of-Scope | Salutations, spam, single emojis, unintelligible text, off-topic conversational fillers. | Canned polite response or request for clarification. |

---

## 4. Escalation Policy

An example is marked `escalate` if ANY of the following criteria are met:
1. **Safety & Hazard**: Any mention of device smoking, burning, swelling battery, puffy back, screen lifting, or physical injury.
2. **Legal / Regulatory Risk**: Threats of litigation, lawyer consultation, or formal complaints to regulators (FTC, BBB).
3. **Explicit Agent Request**: Customer explicitly demands a manager, human representative, or supervisor.
4. **PII / Authentication Boundary**: Issues requiring handling of credentials, bank cards, or 2FA overrides.
5. **High-Value Financial Disputes**: Accusations of fraud, recurring unauthorized charges, or refusal of automatic refund.
6. **Multi-Turn Frustration**: Inquiries showing 3+ failed automated resolution attempts.

All other routine troubleshooting (reboots, network resets, setting adjustments, compatibility checks) are marked `auto_handle`.

---

## 5. Human Validation for LLM-as-Judge

To assess judge reliability on actual agent-generated replies:
- 45 agent-generated replies across held-out golden cases were evaluated on the 5 dimensions on a 1–5 integer scale.
- Scores are recorded in `data/judge/human_review.csv` and imported to `data/judge/human_ratings.json`.
- Provenance is explicitly single human engineer (`human_single_annotator`), avoiding fabricated second raters.
- Judge calibration is evaluated via **Mean Absolute Error (MAE)**, **exact agreement rate**, **agreement within $\pm 1$ point**, **Spearman rank correlation ($\rho$)**, and **Pearson correlation ($r$)**.
- Full results are reported transparently in `results/benchmark_metrics.json` and `REPORT.md`.
