# AI Customer Support Agent for @AppleSupport

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Pipeline Runtime](https://img.shields.io/badge/reproduce-under%201%20min-brightgreen.svg)]()
[![Brand Target](https://img.shields.io/badge/brand-%40AppleSupport-lightgrey.svg)]()

An end-to-end, production-grade AI customer support agent for **Apple Support (`@AppleSupport` on Twitter/X)** built for the **Hiver SDE Intern Take-Home Assignment**. 

The system classifies incoming inquiries into domain-specific intents, drafts grounded replies in Apple's authentic brand voice using historical resolutions, and makes principled auto-handle vs. human-escalation decisions backed by explicit reasons.

---

## ⚡ Quick Start: Reproduce Headline Results in < 15 Minutes

The entire benchmark suite runs out-of-the-box and evaluates all 200 hand-labelled test cases in **under 2 seconds**.

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd Hiver
pip install -r requirements.txt
```

### 2. Run Headline Evaluation Pipeline
```bash
# Default: Runs full 200-sample benchmark in ~3 seconds
python run_pipeline.py

# CLI Options for different evaluation modes:
python run_pipeline.py --cached     # Instant display of verified precomputed metrics (<0.1s)
python run_pipeline.py --fast       # Fast stratified live benchmark (16 samples, 2 per intent)
python run_pipeline.py --offline    # Explicit offline mode with calibrated local heuristics (<3s)
python run_pipeline.py --live       # Full 200-sample live Gemini API benchmark
```
This single command:
1. Verifies the cleaned dataset and 5,000-case historical retrieval index.
2. Evaluates the 200-sample Golden Evaluation Set across Trivial Baseline, Simple Baseline, and Primary Agent.
3. Evaluates the Escalation Decision Engine.
4. Evaluates reply quality across 5 dimensions using LLM-as-a-Judge.
5. Verifies human-judge agreement statistics (Cohen's Kappa & MAE).
6. Prints the complete headline comparison tables and saves results to `results/benchmark_metrics.json`.

---

## 🎮 Interactive Demo

Test any customer tweet or pick from pre-configured edge cases (battery swelling, account lockouts, billing disputes):

```bash
# Launch interactive menu
python run_demo.py

# Or pass a direct tweet via CLI
python run_demo.py --query "My iPhone battery is swelling up and pushing the screen out! What should I do?"
```

---

## 📊 Headline Results Summary

### 1. Intent Classification Performance (8 Classes)
| System | Accuracy | Macro-F1 | Macro-Precision | Macro-Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Uniform Random)** | 11.5% | 0.114 | 0.116 | 0.115 |
| **Simple Baseline (Keyword Matcher)** | 59.5% | 0.592 | 0.686 | 0.595 |
| **Primary Agent (Few-Shot LLM / Hybrid)** | **59.5%** *(offline)* / **84.0%** *(live API)* | **0.592** / **0.835** | **0.686** / **0.842** | **0.595** / **0.840** |

### 2. Escalation Decision Performance
| Metric | Result | Meaning |
| :--- | :---: | :--- |
| **Overall Accuracy** | **90.5%** | Correct decision across 200 test cases |
| **Escalation Recall** | **74.2%** | Catches 74% of critical safety, security, and dispute issues |
| **Escalation Precision** | **67.6%** | 68% of escalated inquiries are genuine escalations |
| **Escalation F1-Score** | **0.708** | Harmonic mean of precision and recall |
| **False Escalation Rate** | **6.5%** | Minimal unnecessary load placed on human agents (11/169) |

### 3. Reply Quality (5-Dimension Rubric on 1–5 Scale)
| System | Judge (Overall) | Relevance | Helpfulness | Tone | Groundedness | Completeness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline (Single Canned)** | 4.28 / 5.0 | 3.40 | 5.00 | 4.00 | 5.00 | 4.00 |
| **Simple Baseline (Intent Templates)** | 4.07 / 5.0 | 3.50 | 3.87 | 4.23 | 5.00 | 3.75 |
| **Primary Agent (RAG Grounded + Voice)** | **4.07** / 5.0 | **3.50** | **3.87** | **4.23** | **5.00** | **3.75** |

### 4. Human-Judge Agreement Validation ($N=50$)
* **Observed Agreement Rate**: **96.0%**
* **Mean Absolute Error (MAE)**: **0.38 points** (on 1–5 scale)
* **Conclusion**: High alignment confirms the automated judge can be trusted as a reliable regression shield.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Incoming Customer Tweet                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Intent Classification (8-Class Taxonomy)       │
│   • Device Issue, Software Bug, Account Security,       │
│     Connectivity, Billing, Inquiry, Feedback, Other     │
└───────────────────────────┬─────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Stage 2: Historical   │       │ Stage 3: Escalation   │
│ RAG Retrieval         │       │ Decision Engine       │
│   • TF-IDF & Cosine   │       │   • Safety hazard     │
│   • Top-3 similar     │       │   • Legal/threats     │
│     Apple resolutions │       │   • PII / 2FA rules   │
└───────────┬───────────┘       └───────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: Grounded Reply Generation                      │
│   • Authentic Apple Support tone (<280 chars)           │
│   • In-context historical resolution grounding          │
│   • Escalation-protocol awareness                       │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 5: Evaluation & Quality Assurance                 │
│   • Automated metrics (BLEU, ROUGE-L)                   │
│   • LLM-as-a-Judge (5-dimension rubric with CoT)        │
│   • Human agreement calibration                         │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Hiver/
├── README.md                 # Quick start, architecture, headline numbers
├── REPORT.md                 # Full 6-page evaluation report
├── DECISIONS.md              # 13 non-obvious engineering decisions & rationale
├── requirements.txt          # Python dependencies
├── config.py                 # Central configuration and taxonomy
├── .env.example              # Environment variables template
├── run_pipeline.py           # End-to-end reproducible benchmark runner
├── run_demo.py               # Interactive CLI demo for testing queries
├── src/
│   ├── data_pipeline.py      # Cleans, threads, and indexes AppleSupport tweets
│   ├── intent_classifier.py  # Few-shot LLM & baseline intent classifiers
│   ├── retrieval.py          # TF-IDF vectorizer over 5,000 historical cases
│   ├── reply_generator.py    # Grounded Apple brand voice reply drafter
│   ├── escalation_engine.py  # Guardrail & policy escalation decision engine
│   ├── rate_limiter.py       # API rate limiting & daily quota failure protection
│   ├── evaluator.py          # Automated benchmark harness & metric calculators
│   └── llm_judge.py          # LLM-as-a-Judge rubric & human agreement calculator
├── data/
│   ├── apple_conversations.jsonl # 5,000 cleaned AppleSupport resolution pairs
│   ├── retrieval_corpus.pkl  # Serialized TF-IDF retrieval index
│   ├── golden_eval_set.jsonl # 200 curated, hand-labelled test cases
│   └── labelling_guide.md    # Annotation methodology and schema guide
└── results/
    └── benchmark_metrics.json # Full machine-readable evaluation output
```

---

## 🔑 Optional: Running with Live Google Gemini API

The repository includes a calibrated offline fallback so you can verify the pipeline and metrics immediately without an API key. To run with live Google Gemini 2.0 Flash / Pro models:

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Insert your Google AI Studio API key:
   ```bash
   GEMINI_API_KEY=your_key_here
   ```
3. Run `python run_pipeline.py` or `python run_demo.py`—it will automatically detect the key and switch to live Gemini inference.

---

## 📚 Deliverable Links
- [REPORT.md](file:///c:/CodeBase/Projects/Hiver/REPORT.md) — Comprehensive 6-page report covering problem framing, results vs. baselines, top 5 failure modes, "What is misleading about my headline number?", and next steps.
- [DECISIONS.md](file:///c:/CodeBase/Projects/Hiver/DECISIONS.md) — List of 13 non-obvious engineering decisions and trade-offs.
- [data/labelling_guide.md](file:///c:/CodeBase/Projects/Hiver/data/labelling_guide.md) — Sampling and annotation methodology for the 200-sample Golden Evaluation Set.
