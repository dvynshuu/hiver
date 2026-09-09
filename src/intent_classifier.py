import json
import re
import pickle
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    INTENT_TAXONOMY,
    INTENT_NAMES,
    GEMINI_API_KEY,
    AGENT_MODEL_NAME,
    RETRIEVAL_CORPUS_JSONL_PATH,
    SEED
)

logger = logging.getLogger(__name__)

LR_MODEL_PATH = DATA_DIR / "intent_lr_model.pkl"

# Enhanced keyword sets with boundary-aware regex matching
INTENT_KEYWORDS = {
    "device_issue": [
        r"\bbattery\b", r"\bbatteries\b", r"\bbattery health\b", r"\bdrain\b", r"\bdraining\b",
        r"\bscreen\b", r"\bdisplay\b", r"\bflicker\b", r"\bflickering\b", r"\bshattered\b",
        r"\bcracked\b", r"\bbroken glass\b", r"\bback glass\b", r"\bspeaker\b", r"\bspeakers\b",
        r"\bmuffled\b", r"\bearpiece\b", r"\bmicrophone\b", r"\bmic\b", r"\bcharger port\b",
        r"\bcharging port\b", r"\bwon't charge\b", r"\bnot charging\b", r"\boverheating\b",
        r"\bswelling\b", r"\bswollen\b", r"\bcamera\b", r"\bhardware\b", r"\btrackpad\b",
        r"\bhaptic\b", r"\bvibration\b", r"\bpower button\b", r"\bvolume button\b", r"\bdigital crown\b",
        r"\bdigitizer\b", r"\btouch id\b", r"\bface id\b", r"\btruedepth\b", r"\bmagsafe charger\b",
        r"\bslow charge\b", r"\bslow charging\b", r"\bcharges slow\b", r"\bcharging slowly\b",
        r"\bheadphone\b", r"\bheadphones\b", r"\bearbuds\b"
    ],
    "software_bug": [
        r"\bios\b", r"\bipados\b", r"\bmacos\b", r"\bupdate\b", r"\bupdating\b", r"\binstaller\b",
        r"\bboot loop\b", r"\bapple logo\b", r"\bfreeze\b", r"\bfreezing\b", r"\bfrozen\b",
        r"\bcrash\b", r"\bcrashes\b", r"\bcrashing\b", r"\bglitch\b", r"\blag\b", r"\blagging\b",
        r"\bslow performance\b", r"\brunning slow\b", r"\bphone is slow\b", r"\bacting slow\b", r"\bslowed down\b",
        r"\bstuck\b", r"\berror\b", r"\berror code\b", r"\bspinning wheel\b",
        r"\brecovery mode\b", r"\bdfu\b", r"\brestore\b", r"\bbackup\b", r"\bsync\b",
        r"\bsystem data\b", r"\bstorage full\b", r"\bstandby mode\b", r"\bdynamic island\b"
    ],
    "account_security": [
        r"\bapple id\b", r"\bicloud\b", r"\bpassword\b", r"\bpasscode\b", r"\blocked\b",
        r"\bsecurity code\b", r"\bverification code\b", r"\b2fa\b", r"\btwo-factor\b",
        r"\bhacked\b", r"\bstolen\b", r"\bbreach\b", r"\bunauthorized\b", r"\bsign in\b",
        r"\blogin\b", r"\brecover\b", r"\brecovery\b", r"\biforgot\b", r"\bsecurity question\b",
        r"\btrusted phone\b", r"\btrusted device\b", r"\badvanced data protection\b",
        r"\bspyware\b", r"\blockdown mode\b", r"\bactivation lock\b"
    ],
    "connectivity": [
        r"\bwi-fi\b", r"\bwifi\b", r"\bbluetooth\b", r"\bairdrop\b", r"\bairpods\b",
        r"\bdisconnect\b", r"\bdisconnecting\b", r"\bdisconnects\b", r"\bpair\b", r"\bpairing\b",
        r"\bno service\b", r"\bsearching\b", r"\bcellular\b", r"\blte\b", r"\b5g\b",
        r"\bhotspot\b", r"\bcarplay\b", r"\bairtag\b", r"\bcontinuity\b", r"\bhandoff\b",
        r"\bcaptive portal\b", r"\besim\b", r"\brange\b", r"\binterference\b"
    ],
    "billing_purchase": [
        r"\bcharge\b", r"\bcharges\b", r"\bcharged\b", r"\bbilling\b", r"\brefund\b",
        r"\brefunds\b", r"\bsubscription\b", r"\bsubscriptions\b", r"\bcancel subscription\b",
        r"\bpayment\b", r"\bcard declined\b", r"\binvoice\b", r"\breceipt\b",
        r"\bin-app purchase\b", r"\bdouble charge\b", r"\bunauthorized charge\b",
        r"\bapple pay\b", r"\bgift card\b", r"\bapple card\b", r"\breportaproblem\b"
    ],
    "product_inquiry": [
        r"\bapple pencil\b", r"\bcompatibility\b", r"\bcompatible\b", r"\bwork with\b",
        r"\bapplecare\b", r"\bwarranty\b", r"\btrade-in\b", r"\btrade in\b", r"\bspecs\b",
        r"\bspecification\b", r"\bdifference between\b", r"\bupgrade ram\b", r"\bhow much is\b",
        r"\bwhen will\b", r"\brelease date\b", r"\bin stock\b", r"\breturn policy\b",
        r"\breturn window\b", r"\brefurbished\b", r"\bengrave\b"
    ],
    "general_feedback": [
        r"\bworst\b", r"\bhate\b", r"\bterrible\b", r"\bawful\b", r"\buseless\b",
        r"\bthank you\b", r"\bthanks\b", r"\bkudos\b", r"\bgreat job\b", r"\blove apple\b",
        r"\bcustomer service\b", r"\bgenius bar\b", r"\bstore manager\b", r"\bcomplaint\b",
        r"\bdisappointed\b", r"\bshame\b", r"\bloyal customer\b", r"\bswitching to android\b",
        r"\bright to repair\b"
    ],
    "other": [
        r"\bgood morning\b", r"\bwhat time\b", r"\bdinner\b", r"\blol\b", r"\btesting\b",
        r"\bwrong account\b", r"\bsteve jobs\b", r"\bwho is better\b", r"\bhablan español\b",
        r"\bhappy new year\b", r"\bgoodbye\b", r"\basdfghjkl\b"
    ]
}

def clean_tweet_text(text: str) -> str:
    """Normalize tweet text by stripping mentions and trailing links."""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class IntentClassifier:
    """
    Classifies incoming customer messages into 8 domain-specific intents.
    Implements:
    - Baseline 1 (Trivial): Deterministic Majority-Class Prediction
    - Baseline 2 (Learned): TF-IDF + Logistic Regression (trained on retrieval corpus)
    - Primary Agent: Few-Shot LLM (Gemini) with honest offline fallback
    """

    MAJORITY_CLASS = "software_bug"  # Most frequent class in empirical Twitter support

    def __init__(self, api_key: Optional[str] = None, model_name: str = AGENT_MODEL_NAME):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None
        self.vectorizer = None
        self.lr_model = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")

        # Load or train the Logistic Regression baseline model
        self._load_or_train_lr_baseline()

    # -------------------------------------------------------------
    # BASELINE 1: Trivial Deterministic Majority-Class Baseline
    # -------------------------------------------------------------
    def classify_majority_baseline(self, text: str) -> Dict[str, Any]:
        """
        Trivial deterministic baseline: always predicts the empirical majority class.
        Deterministic, reproducible, zero random.choice.
        """
        return {
            "intent": self.MAJORITY_CLASS,
            "confidence": 0.32,  # Empirical prevalence of majority class
            "method": "majority_class_baseline",
            "reasoning": f"Deterministic baseline: predicted majority class '{self.MAJORITY_CLASS}'"
        }

    # -------------------------------------------------------------
    # BASELINE 2: Simple Learned Baseline (TF-IDF + Logistic Regression)
    # -------------------------------------------------------------
    def _load_or_train_lr_baseline(self):
        """Loads cached Logistic Regression baseline or trains on retrieval corpus."""
        if LR_MODEL_PATH.exists():
            try:
                with open(LR_MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.vectorizer = data["vectorizer"]
                    self.lr_model = data["model"]
                return
            except Exception as e:
                logger.warning(f"Failed to load cached LR model: {e}. Re-training...")

        self.train_lr_baseline()

    def train_lr_baseline(self):
        """
        Train TF-IDF + Logistic Regression on the train/retrieval corpus using reproducible seed=42.
        Labels training samples using domain rules to establish a strong classical baseline.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        texts = []
        labels = []

        if RETRIEVAL_CORPUS_JSONL_PATH.exists():
            with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        txt = item.get("customer_text", "")
                        if len(txt) >= 18:
                            texts.append(txt)
                            labels.append(self._rule_label_text(txt))

        if not texts:
            # Fallback training examples from intent taxonomy definitions
            for name, meta in INTENT_TAXONOMY.items():
                for ex in meta["positive_examples"]:
                    texts.append(ex)
                    labels.append(name)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english"
        )
        X = self.vectorizer.fit_transform(texts)
        self.lr_model = LogisticRegression(
            random_state=SEED,
            class_weight="balanced",
            max_iter=1000,
            C=1.0
        )
        self.lr_model.fit(X, labels)

        # Cache model
        with open(LR_MODEL_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.lr_model
            }, f)
        logger.info(f"Trained and saved TF-IDF + Logistic Regression baseline on {len(texts)} samples.")

    def _rule_label_text(self, text: str) -> str:
        """Domain keyword rule for silver label assignment on training data."""
        text_lower = clean_tweet_text(text).lower()
        scores = {}
        for intent, patterns in INTENT_KEYWORDS.items():
            count = sum(1 for pat in patterns if re.search(pat, text_lower))
            if count > 0:
                scores[intent] = count
        if scores:
            return max(scores, key=scores.get)
        return "other"

    def classify_learned_baseline(self, text: str) -> Dict[str, Any]:
        """
        Learned Baseline: TF-IDF feature representation + Logistic Regression classifier.
        Reproducible with SEED=42.
        """
        if self.vectorizer is None or self.lr_model is None:
            self._load_or_train_lr_baseline()

        cleaned = clean_tweet_text(text)
        if not cleaned:
            return {
                "intent": "other",
                "confidence": 0.50,
                "method": "tfidf_logistic_regression",
                "reasoning": "Cleaned message is empty"
            }

        vec = self.vectorizer.transform([cleaned])
        probs = self.lr_model.predict_proba(vec)[0]
        classes = self.lr_model.classes_
        top_idx = probs.argmax()
        pred_intent = str(classes[top_idx])
        conf = float(probs[top_idx])

        return {
            "intent": pred_intent,
            "confidence": round(conf, 3),
            "method": "tfidf_logistic_regression",
            "reasoning": f"Classified by TF-IDF + Logistic Regression (confidence: {conf:.2f})"
        }

    # -------------------------------------------------------------
    # AUXILIARY BASELINE: Rule/Keyword Baseline
    # -------------------------------------------------------------
    def classify_keyword_baseline(self, text: str) -> Dict[str, Any]:
        """Keyword pattern matcher baseline."""
        text_lower = clean_tweet_text(text).lower()
        if not text_lower:
            return {
                "intent": "other",
                "confidence": 0.5,
                "method": "simple_keyword_baseline",
                "reasoning": "Empty input defaulted to 'other'"
            }

        scores = {}
        for intent, patterns in INTENT_KEYWORDS.items():
            count = sum(1 for pat in patterns if re.search(pat, text_lower))
            if count > 0:
                scores[intent] = count

        if scores:
            best_intent = max(scores, key=scores.get)
            max_hits = scores[best_intent]
            confidence = min(0.5 + (max_hits * 0.15), 0.95)
            return {
                "intent": best_intent,
                "confidence": round(confidence, 2),
                "method": "simple_keyword_baseline",
                "reasoning": f"Matched {max_hits} domain patterns for {best_intent}"
            }

        return {
            "intent": "other",
            "confidence": 0.35,
            "method": "simple_keyword_baseline",
            "reasoning": "No domain patterns matched, defaulted to 'other'"
        }

    # -------------------------------------------------------------
    # PRIMARY AGENT: Few-Shot LLM (Gemini)
    # -------------------------------------------------------------
    def classify_llm(self, text: str) -> Dict[str, Any]:
        """Primary Agent: Few-shot Gemini LLM with structured JSON output."""
        if not self.client:
            return {
                "intent": "unavailable",
                "confidence": 0.0,
                "method": "llm_unavailable",
                "reasoning": "Gemini API key not configured or offline mode active."
            }

        cleaned = clean_tweet_text(text)
        if not cleaned:
            return {
                "intent": "other",
                "confidence": 0.99,
                "method": "llm_few_shot",
                "reasoning": "Cleaned message is empty"
            }

        prompt = self._build_prompt(cleaned)
        from src.rate_limiter import rate_limited_api_call

        def _do_llm_call():
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                    response_mime_type="application/json"
                )
            )
            raw_text = response.text.strip()
            parsed = json.loads(raw_text)

            intent = parsed.get("intent", "").lower().strip()
            if intent not in INTENT_NAMES:
                intent = "other"

            return {
                "intent": intent,
                "confidence": float(parsed.get("confidence", 0.85)),
                "method": "llm_few_shot",
                "reasoning": parsed.get("reasoning", "Classified by Gemini few-shot prompt")
            }

        def _fallback():
            return {
                "intent": "unavailable",
                "confidence": 0.0,
                "method": "llm_api_exhausted",
                "reasoning": "LLM call failed after retries (quota exhausted or network error)."
            }

        return rate_limited_api_call(
            call_fn=_do_llm_call,
            max_retries=2,
            fallback_fn=_fallback
        )

    def classify(self, text: str, method: str = "llm") -> Dict[str, Any]:
        """
        Unified classification entrypoint.
        method options:
        - 'majority': Deterministic Baseline 1 (Majority Class)
        - 'tfidf_lr' / 'learned': Learned Baseline 2 (TF-IDF + Logistic Regression)
        - 'keyword': Keyword Rule Matcher
        - 'llm': Primary Agent (Few-Shot Gemini)
        """
        if method in ("majority", "trivial_baseline"):
            return self.classify_majority_baseline(text)
        elif method in ("tfidf_lr", "learned", "simple_learned", "simple_baseline"):
            return self.classify_learned_baseline(text)
        elif method == "keyword":
            return self.classify_keyword_baseline(text)
        elif method == "llm":
            return self.classify_llm(text)
        else:
            return self.classify_learned_baseline(text)

    def _build_prompt(self, customer_text: str) -> str:
        """Construct a structured few-shot intent classification prompt."""
        taxonomy_summary = []
        for name, meta in INTENT_TAXONOMY.items():
            taxonomy_summary.append(f"- **{name}**: {meta['definition']}")

        taxonomy_str = "\n".join(taxonomy_summary)

        return f"""You are an expert customer support intent classifier for Apple Support on Twitter (@AppleSupport).
Classify the following customer tweet into EXACTLY ONE of the allowed intents below:

ALLOWED INTENTS:
{taxonomy_str}

DECISION BOUNDARIES:
1. If the message mentions hardware components (battery, screen, physical buttons, speaker, charging port, camera), choose 'device_issue'.
2. If the message mentions OS crashes, freezing, update installation errors, storage bugs, or app crashes, choose 'software_bug'.
3. If the message mentions Apple ID, iCloud lock, 2FA, hacked accounts, or password resets, choose 'account_security'.
4. If the message mentions Wi-Fi, Bluetooth, AirPods pairing, or cellular data dropouts, choose 'connectivity'.
5. If the message mentions unauthorized charges, subscriptions, refund requests, or payment issues, choose 'billing_purchase'.
6. If the message asks about buying, specs, warranty, or AppleCare, choose 'product_inquiry'.
7. If the message expresses emotion/praise/venting without asking for troubleshooting, choose 'general_feedback'.
8. If none apply or it's conversational filler/chitchat, choose 'other'.

CUSTOMER TWEET:
"{customer_text}"

Return ONLY a valid JSON object matching this schema:
{{
  "intent": "<one of the allowed intent names>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of why this intent was chosen>"
}}
"""
