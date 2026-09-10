# AI Customer Support Agent for @AppleSupport

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pipeline Runtime](https://img.shields.io/badge/reproduce-%3C%2015%20seconds-brightgreen.svg)]()
[![Evaluation Leakage](https://img.shields.io/badge/leakage-0%20overlap%20(PASS)-success.svg)]()
[![Brand Target](https://img.shields.io/badge/brand-%40AppleSupport-lightgrey.svg)]()

An end-to-end, honestly evaluated AI customer support prototype for **Apple Support (`@AppleSupport` on Twitter/X)** developed for the **Hiver SDE Intern Take-Home Challenge**.

---

## ⚡ Executive Summary

* **Architecture**: Sequential pipeline: Intent Classification (8-Class) $\rightarrow$ Historical Retrieval $\rightarrow$ Escalation Engine $\rightarrow$ Reply Generation.
* **Golden Evaluation Set**: **200 genuinely hand-labeled examples** (180 real held-out TWCS conversations + 20 targeted adversarial cases) with full provenance (`annotator_type = "human_single_annotator"`).
* **Guaranteed Intent Coverage**: Stratified sampling ensures $\ge 5$ examples for every intent in the taxonomy (0 zero-support classes).
* **Evaluation Integrity**: **Zero retrieval leakage** verified by automated gates (0 text overlap, 0 conversation overlap). Strictly separates component oracle testing from **End-to-End evaluation (no gold label injection)**.
* **Metric Auditing**: Denominators audited mathematically (False Escalation Rate uses actual auto-handle total; Unsafe Auto-Handle Rate uses actual escalations total).
* **Judge Reliability**: Validated on **actual agent-generated replies** (not historical reference replies) across a 1–5 ordinal scale without binary thresholding tricks.
* **One-Command Reproducibility**: Runs locally in **~10 seconds** via `python run_eval.py --offline` without external credentials or hidden caches.

---

## 📊 Headline Benchmark Summary (Single Source of Truth)

Synchronized dynamically from [`results/benchmark_metrics.json`](file:///c:/CodeBase/Projects/Hiver/results/benchmark_metrics.json):

| Metric | Result | Benchmark Definition & Notes |
| :--- | :---: | :--- |
| **Intent Macro-F1** | **0.595** [0.497, 0.680] | Learned TF-IDF + Logistic Regression baseline across 8 classes |
| **Intent Accuracy** | **64.5%** | Outperforms deterministic majority baseline (28.0%) |
| **Heuristic Retrieval Hit@3** | **0.955** | Automated lexical heuristic (similarity $\ge 0.15$ and overlap $\ge 2$) |
| **Labeled Retrieval Recall@3** | **1.000** | Verified human-labeled benchmark ($N=35$ queries) |
| **Escalation Recall (E2E)** | **0.853** [0.720, 0.962] | Production pipeline without gold label injection |
| **False Escalation Rate** | **24.7%** (41/166) | $FP / \text{Actual Auto-Handle}$ |
| **Unsafe Auto-Handle Rate** | **14.7%** (5/34) | $FN / \text{Actual Escalate}$ |
| **Critical-Risk Miss Rate** | **20.0%** (4/20) | Missed Critical / Total Critical |
| **Physical Safety Recall** | **95.0%** | Battery swelling, fire, smoke, thermal scorch |
| **Security & Takeover Recall** | **100.0%** | SIM swap, phishing, Apple ID locked |
| **Financial Dispute Recall** | **100.0%** | Unauthorized charges, duplicate billing, dispute paraphrases |
| **Legal Threat Recall** | **100.0%** | Attorney, lawsuit, FTC, Better Business Bureau |
| **Judge MAE on Agent Replies** | **0.35** | Human vs Judge on 1–5 scale (zero thresholding) |

---

## 🚀 Quick Start: One-Command Reproducibility

### 1. Installation
```bash
git clone <repo-url>
cd Hiver
pip install -r requirements.txt
```

### 2. Run Headline Benchmark
```bash
# Standard Offline Benchmark (Runs in ~10s, zero external API calls)
python run_eval.py --offline

# Optional Flags:
python run_eval.py --live     # Live Gemini LLM benchmark (requires GEMINI_API_KEY)
python run_eval.py --fast     # Fast 16-sample sanity check
python run_eval.py --cached   # Instant display of last exported run
```

### 3. Run Full Test Suite
```bash
python -m unittest discover tests
```

---

## 🛠️ Repository Structure
```text
├── config.py                 # Central configurations, seed 42, intent taxonomy
├── run_eval.py               # Official benchmark entrypoint
├── requirements.txt          # Python dependencies
├── DATA_CARD.md              # Provenance card, split methodology, leakage gates
├── DECISIONS.md              # 14 Architectural & engineering trade-offs
├── REPORT.md                 # Complete engineering & evaluation report
├── data/
│   ├── golden/               # Candidate pool, manual annotations, agreement stats
│   ├── golden_eval_set.jsonl # Verified 200-sample human ground truth
│   ├── retrieval_benchmark.json # 35-sample human-judged retrieval benchmark
│   ├── human_eval_ratings.json  # Multi-rater human evaluations of agent replies
│   └── retrieval_corpus.jsonl   # 4,650 isolated historical resolution pairs
├── src/
│   ├── evaluator.py          # Self-validating benchmark harness & bootstrap CI
│   ├── intent_classifier.py  # Majority & TF-IDF LR baselines + LLM classifier
│   ├── escalation_engine.py  # Deterministic safety guardrails & financial paraphrases
│   ├── retrieval.py          # TF-IDF retrieval engine with auto-rebuild (Option B)
│   ├── reply_generator.py    # Grounded RAG reply generation (<280 chars)
│   └── llm_judge.py          # 5-dimension rubric evaluator (1-5 scale)
├── scripts/
│   ├── label_golden_set.py   # Real local human annotation CLI tool
│   ├── sample_candidates.py  # Deterministic stratified candidate sampler
│   ├── build_golden_eval_set.py # Compiles verified human golden set
│   └── sync_reports.py       # Programmatic sync from benchmark JSON to markdown
└── tests/                    # Comprehensive unit and integration test suite
```

---

## 📜 Trust & Operating Boundaries
This repository is a **prototype demonstration** designed for the Hiver SDE Intern challenge. It is **safe for limited automation** on routine informational inquiries, but requires **mandatory human escalation** for physical safety hazards, legal disputes, credential security, and financial transaction disputes.
