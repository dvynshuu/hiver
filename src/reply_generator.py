import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GEMINI_API_KEY, AGENT_MODEL_NAME
from src.retrieval import HistoricalRetrievalEngine


logger = logging.getLogger(__name__)

# Canned template responses for Simple Baseline
INTENT_TEMPLATES = {
    "device_issue": "We'd like to help with your device issue. Try restarting your device, check for updates in Settings > General > Software Update, and ensure your battery health is normal. Reach out if you still need assistance!",
    "software_bug": "We understand system glitches can be frustrating. Please ensure your device is backed up and updated to the latest software release. If the issue persists, let us know what iOS version you are on.",
    "account_security": "Your account security is our top priority. Please visit https://iforgot.apple.com to reset your password, or review your trusted devices. For account-specific security, please DM us.",
    "connectivity": "Let's get you connected again. Try toggling Airplane Mode on and off, resetting network settings in Settings > General > Transfer or Reset iPhone > Reset Network Settings, or re-pairing your device.",
    "billing_purchase": "We can help you review your purchases. You can check your purchase history and request refunds directly at https://reportaproblem.apple.com. Feel free to DM us if you have questions.",
    "product_inquiry": "Thanks for your interest! You can compare specs and check warranty/AppleCare coverage at https://checkcoverage.apple.com. Let us know if you need specific details.",
    "general_feedback": "Thank you for sharing your feedback with us. We are always striving to improve our products and services for our customers.",
    "other": "Thanks for reaching out to Apple Support. How can we help you today? Please feel free to share more details so we can assist."
}

class ReplyGenerator:
    """
    Drafts replies grounded in historical AppleSupport resolutions.
    Implements:
    - Primary Agent: RAG + Few-Shot Gemini LLM in AppleSupport brand voice
    - Simple Baseline: Intent-based canned template
    - Trivial Baseline: Generic single canned reply
    """

    def __init__(
        self,
        retrieval_engine: Optional[HistoricalRetrievalEngine] = None,
        api_key: Optional[str] = None,
        model_name: str = AGENT_MODEL_NAME
    ):
        self.retrieval_engine = retrieval_engine
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client for generator: {e}")

    # -------------------------------------------------------------
    # BASELINE 1: Trivial Baseline
    # -------------------------------------------------------------
    def generate_trivial_baseline(self, text: str) -> Dict[str, Any]:
        """Trivial baseline: one generic response regardless of input."""
        return {
            "reply": "Thanks for reaching out! Please restart your device or visit https://support.apple.com for more info.",
            "method": "trivial_baseline",
            "grounded_cases": []
        }

    # -------------------------------------------------------------
    # BASELINE 2: Simple Template Baseline
    # -------------------------------------------------------------
    def generate_simple_baseline(
        self,
        text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> Dict[str, Any]:
        """Simple baseline: rule/intent-mapped canned reply with escalation awareness."""
        if escalation_decision == "escalate":
            if "safety" in escalation_reason.lower() or any(w in text.lower() for w in ["swell", "smoke", "fire", "exploded"]):
                reply = "Your safety is our top priority. Please stop using and charging the device immediately, disconnect it from power, and bring it to an Apple Store or contact AppleCare directly for priority safety inspection."
            elif intent == "account_security":
                reply = "Account security is our top priority. Please visit https://iforgot.apple.com immediately to secure your Apple ID, or send us a Direct Message so our account specialists can assist."
            elif "human" in escalation_reason.lower() or "supervisor" in text.lower():
                reply = "We'd be glad to connect you with a Senior Specialist. Please select this link to join us in a Direct Message with your details: https://twitter.com/messages/compose?recipient_id=AppleSupport"
            else:
                reply = "Let's take a closer look into this together. Please join us in a Direct Message with your device details and we'll escalate this to the appropriate specialist team: https://twitter.com/messages/compose?recipient_id=AppleSupport"
        else:
            reply = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["other"])

        return {
            "reply": reply,
            "method": "simple_template_baseline",
            "intent_used": intent,
            "grounded_cases": []
        }


    # -------------------------------------------------------------
    # PRIMARY AGENT: RAG + LLM in Apple Brand Voice
    # -------------------------------------------------------------
    def generate_reply(
        self,
        customer_text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = "",
        method: str = "rag_llm"
    ) -> Dict[str, Any]:
        """
        Generate grounded customer reply.
        method: 'rag_llm', 'simple_baseline', or 'trivial_baseline'
        """
        if method == "trivial_baseline":
            return self.generate_trivial_baseline(customer_text)
        elif method == "simple_baseline":
            return self.generate_simple_baseline(customer_text, intent, escalation_decision, escalation_reason)

        # Retrieve historical cases
        retrieved_cases = []
        retrieval_context = "No historical cases available."
        if self.retrieval_engine:
            retrieved_cases = self.retrieval_engine.retrieve(customer_text, top_k=3)
            retrieval_context = self.retrieval_engine.format_retrieval_context(retrieved_cases)

        # If LLM client unavailable, use template baseline with retrieved knowledge note
        if not self.client:
            fallback = self.generate_simple_baseline(customer_text, intent, escalation_decision, escalation_reason)
            fallback["grounded_cases"] = retrieved_cases
            fallback["note"] = "Gemini API key not configured; returned template baseline."
            return fallback

        prompt = self._build_generation_prompt(
            customer_text, intent, retrieval_context, escalation_decision, escalation_reason
        )


        from src.rate_limiter import rate_limited_api_call

        def _do_llm_call():
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )

            reply_text = response.text.strip()
            # Clean enclosing quotes if any
            if reply_text.startswith('"') and reply_text.endswith('"'):
                reply_text = reply_text[1:-1].strip()

            return {
                "reply": reply_text,
                "method": "rag_llm",
                "intent": intent,
                "grounded_cases": [
                    {
                        "similarity": c.get("similarity_score", 0),
                        "historical_query": c.get("customer_text", ""),
                        "historical_reply": c.get("support_reply", "")
                    }
                    for c in retrieved_cases
                ]
            }

        def _fallback():
            fallback = self.generate_simple_baseline(customer_text, intent)
            fallback["grounded_cases"] = retrieved_cases
            fallback["fallback_reason"] = "LLM call failed after retries"
            return fallback

        return rate_limited_api_call(
            call_fn=_do_llm_call,
            max_retries=3,
            fallback_fn=_fallback
        )

    def _build_generation_prompt(
        self,
        customer_text: str,
        intent: str,
        retrieval_context: str,
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> str:
        escalation_guidance = ""
        if escalation_decision == "escalate":
            escalation_guidance = f"""
SPECIAL ESCALATION PROTOCOL:
This inquiry is flagged for HUMAN ESCALATION. Reason: {escalation_reason}
Do NOT provide standard routine troubleshooting if safety or credentials are at risk.
Instead:
- If safety risk (swelling, fire, smoke): Instruct them to stop using and charging the device immediately, disconnect from power, and contact AppleCare or an Apple Store.
- If security/lockout: Direct them to iforgot.apple.com to initiate account recovery, or invite them to DM for secure specialist assistance.
- If legal or supervisor request: Invite them to Direct Message so a Senior Specialist can review their case.
"""

        return f"""You are a customer support specialist for Apple Support on Twitter (@AppleSupport).
Draft a complete, helpful, empathetic reply to the customer's tweet.

REQUIREMENTS:
1. Provide a COMPLETE reply in 2-3 concise sentences (aim for under 280 characters to match Twitter limits). Do NOT leave sentences unfinished.
2. Tone: Warm, empathetic, professional, solution-oriented.
3. Content: Acknowledge the issue, provide the concrete next step or official link (e.g. iforgot.apple.com or reportaproblem.apple.com), and invite them to DM or reply if they need more help.
4. Grounding: Reflect how Apple Support historically resolves this issue.
{escalation_guidance}
HISTORICAL SIMILAR RESOLUTIONS:
{retrieval_context}

DETECTED CUSTOMER INTENT: {intent}
ESCALATION STATUS: {escalation_decision}

CUSTOMER TWEET:
"{customer_text}"

Return ONLY the complete reply text to be tweeted (no markdown, no preamble, no quotes):"""


