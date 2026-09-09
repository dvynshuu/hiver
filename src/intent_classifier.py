import json
import re
import random
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    INTENT_TAXONOMY,
    INTENT_NAMES,
    GEMINI_API_KEY,
    AGENT_MODEL_NAME
)


logger = logging.getLogger(__name__)

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
        r"\bslow charge\b", r"\bslow charging\b", r"\bcharges slow\b", r"\bcharging slowly\b"
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
    - Primary Agent: Few-shot LLM (Gemini)
    - Simple Baseline: Keyword-rule matching + TF-IDF fallback
    - Trivial Baseline: Uniform random sampling
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = AGENT_MODEL_NAME):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None
        
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")

    # -------------------------------------------------------------
    # BASELINE 1: Trivial Random Baseline
    # -------------------------------------------------------------
    def classify_trivial_baseline(self, text: str) -> Dict[str, Any]:
        """Trivial baseline: randomly samples an intent with equal probability."""
        intent = random.choice(INTENT_NAMES)
        return {
            "intent": intent,
            "confidence": 1.0 / len(INTENT_NAMES),
            "method": "trivial_random_baseline",
            "reasoning": "Random assignment (uniform distribution baseline)"
        }

    # -------------------------------------------------------------
    # BASELINE 2: Simple Rule/Keyword Baseline
    # -------------------------------------------------------------
    def classify_simple_baseline(self, text: str) -> Dict[str, Any]:
        """Simple baseline: counts domain keyword/regex pattern overlaps."""
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
            count = 0
            for pat in patterns:
                if re.search(pat, text_lower):
                    count += 1
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
    def classify(self, text: str, method: str = "llm") -> Dict[str, Any]:
        """
        Classify customer query intent.
        method: 'llm', 'simple_baseline', or 'trivial_baseline'
        """
        if method == "trivial_baseline":
            return self.classify_trivial_baseline(text)
        elif method == "simple_baseline":
            return self.classify_simple_baseline(text)

        # Primary LLM classifier
        if not self.client:
            # Fallback to simple baseline if API client is not configured
            result = self.classify_simple_baseline(text)
            result["note"] = "Gemini API key not configured; used rule-based classification."
            return result

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
                    max_output_tokens=1000,
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
            fallback = self.classify_simple_baseline(text)
            fallback["fallback_reason"] = "LLM call failed after retries"
            return fallback

        return rate_limited_api_call(
            call_fn=_do_llm_call,
            max_retries=3,
            fallback_fn=_fallback
        )

    def _build_prompt(self, customer_text: str) -> str:
        """Construct a structured few-shot intent classification prompt."""
        taxonomy_summary = []
        for name, meta in INTENT_TAXONOMY.items():
            taxonomy_summary.append(f"- **{name}**: {meta['description']}")

        taxonomy_str = "\n".join(taxonomy_summary)

        return f"""You are an expert customer support intent classifier for Apple (@AppleSupport).
Classify the following customer tweet into EXACTLY ONE of the allowed intents below:

ALLOWED INTENTS:
{taxonomy_str}

GUIDELINES:
1. If the message mentions hardware (battery, screen, physical buttons, speaker), choose 'device_issue'.
2. If the message mentions OS crashes, freezing, update installation errors, or app bugs, choose 'software_bug'.
3. If the message mentions Apple ID, iCloud lock, 2FA, hacked accounts, or passwords, choose 'account_security'.
4. If the message mentions Wi-Fi, Bluetooth, AirPods pairing, or cellular data, choose 'connectivity'.
5. If the message mentions unauthorized charges, subscriptions, refunds, or payment declined, choose 'billing_purchase'.
6. If the message asks about buying, specs, warranty, or AppleCare, choose 'product_inquiry'.
7. If the message expresses emotion/praise/venting without asking for troubleshooting, choose 'general_feedback'.
8. If none apply, choose 'other'.

CUSTOMER TWEET:
"{customer_text}"

Return ONLY a JSON object with this exact schema:
{{
  "intent": "<one of the allowed intent names>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of why this intent was chosen>"
}}
"""
