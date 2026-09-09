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

# Deterministic Canned Response for Baseline 1
STATIC_CANNED_REPLY = (
    "Thanks for reaching out to Apple Support! To help troubleshoot, please restart your device "
    "and ensure you have the latest software installed. Check https://support.apple.com for guides."
)

# Intent-Mapped Canned Templates for Baseline 2
INTENT_TEMPLATES = {
    "device_issue": (
        "We'd like to help with your device. Try restarting, verify battery health in Settings > Battery, "
        "and inspect ports for debris. Visit https://support.apple.com for repair options or DM us."
    ),
    "software_bug": (
        "We understand glitches are frustrating. Please back up your device and update to the latest iOS/macOS release. "
        "If the issue persists after restarting, send us a DM with your exact OS version."
    ),
    "account_security": (
        "Account security is our top priority. Please visit https://iforgot.apple.com to reset your password or verify trusted devices. "
        "For urgent account assistance, join us in a Direct Message."
    ),
    "connectivity": (
        "Let's get you connected again. Try toggling Airplane Mode, resetting network settings in Settings > General > "
        "Transfer or Reset iPhone > Reset Network Settings, or re-pairing your Bluetooth device."
    ),
    "billing_purchase": (
        "You can review your purchase history and request refunds directly at https://reportaproblem.apple.com. "
        "If you see an unrecognized transaction, please DM us your details so we can assist."
    ),
    "product_inquiry": (
        "Thanks for your interest! You can check device specifications and verify AppleCare coverage at "
        "https://checkcoverage.apple.com. Feel free to DM us if you have specific compatibility questions."
    ),
    "general_feedback": (
        "Thank you for sharing your feedback with Apple Support. We appreciate hearing from our customers "
        "and are always working to improve our products and services."
    ),
    "other": (
        "Thanks for reaching out to Apple Support. How can we help you today? Please reply with more details "
        "about your device and question so we can assist."
    )
}

class ReplyGenerator:
    """
    Drafts replies grounded in historical AppleSupport resolutions.
    Supports 4 distinct modes for ablation analysis:
    1. 'canned': Deterministic single canned reply (Baseline 1)
    2. 'template': Intent-mapped canned template (Baseline 2)
    3. 'llm_no_rag': LLM generation without retrieval context (RAG ablation)
    4. 'rag_llm': Primary Agent - Grounded RAG + Few-Shot LLM in authentic Apple brand voice
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
    # BASELINE 1: Static Canned Response
    # -------------------------------------------------------------
    def generate_canned(self, text: str) -> Dict[str, Any]:
        """Trivial deterministic baseline: single canned response."""
        return {
            "reply": STATIC_CANNED_REPLY,
            "used_evidence": False,
            "evidence_ids": [],
            "confidence": 0.50,
            "method": "canned_baseline"
        }

    # -------------------------------------------------------------
    # BASELINE 2: Intent-Mapped Template Baseline
    # -------------------------------------------------------------
    def generate_template(
        self,
        text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> Dict[str, Any]:
        """Simple baseline: rule-based intent templates with escalation adaptation."""
        if escalation_decision == "escalate":
            t_lower = text.lower()
            if any(w in t_lower for w in ["swell", "smoke", "fire", "exploded", "melted", "puffy"]):
                reply = (
                    "Your safety is our top priority. Please stop using and charging the device immediately, "
                    "disconnect it from power, and visit an Apple Store or DM us for priority safety support."
                )
            elif intent == "account_security" or any(w in t_lower for w in ["hacked", "stolen", "locked out"]):
                reply = (
                    "Your account security is critical. Please visit https://iforgot.apple.com immediately to secure your credentials, "
                    "or send us a Direct Message so our account specialists can assist."
                )
            elif any(w in t_lower for w in ["human", "manager", "supervisor"]):
                reply = (
                    "We'd be glad to connect you with our specialist team. Please join us in a Direct Message with your details: "
                    "https://twitter.com/messages/compose?recipient_id=AppleSupport"
                )
            else:
                reply = (
                    "Let's look into this further together. Please send us a Direct Message with your details so our team can assist: "
                    "https://twitter.com/messages/compose?recipient_id=AppleSupport"
                )
        else:
            reply = INTENT_TEMPLATES.get(intent, INTENT_TEMPLATES["other"])

        return {
            "reply": reply,
            "used_evidence": False,
            "evidence_ids": [],
            "confidence": 0.75,
            "method": "template_baseline",
            "intent_used": intent
        }

    # -------------------------------------------------------------
    # ABLATION: LLM Without Retrieval (No RAG)
    # -------------------------------------------------------------
    def generate_llm_no_rag(
        self,
        customer_text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> Dict[str, Any]:
        """Generates reply using LLM without RAG context (ablation)."""
        if not self.client:
            return {
                "reply": "[LLM Unavailable in Offline Mode]",
                "used_evidence": False,
                "evidence_ids": [],
                "confidence": 0.0,
                "method": "llm_no_rag_offline"
            }

        prompt = self._build_generation_prompt(
            customer_text=customer_text,
            intent=intent,
            retrieval_context="None (Operating without historical retrieval).",
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason
        )

        return self._execute_llm_generation(prompt, used_evidence=False, evidence_ids=[], method_name="llm_no_rag")

    # -------------------------------------------------------------
    # PRIMARY AGENT: Grounded RAG + Few-Shot LLM
    # -------------------------------------------------------------
    def generate_rag_llm(
        self,
        customer_text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = ""
    ) -> Dict[str, Any]:
        """Generates grounded reply using retrieved historical AppleSupport pairs."""
        retrieved_cases = []
        retrieval_context = "No directly matching historical cases found. Follow standard Apple protocols."
        evidence_ids = []

        if self.retrieval_engine:
            retrieved_cases = self.retrieval_engine.retrieve(customer_text, top_k=3)
            if retrieved_cases:
                retrieval_context = self.retrieval_engine.format_retrieval_context(retrieved_cases)
                evidence_ids = [c.get("evidence_id", c.get("pair_id", f"case_{i}")) for i, c in enumerate(retrieved_cases)]

        if not self.client:
            return {
                "reply": "[LLM Unavailable in Offline Mode]",
                "used_evidence": bool(evidence_ids),
                "evidence_ids": evidence_ids,
                "confidence": 0.0,
                "method": "rag_llm_offline"
            }

        prompt = self._build_generation_prompt(
            customer_text=customer_text,
            intent=intent,
            retrieval_context=retrieval_context,
            escalation_decision=escalation_decision,
            escalation_reason=escalation_reason
        )

        return self._execute_llm_generation(prompt, used_evidence=bool(evidence_ids), evidence_ids=evidence_ids, method_name="rag_llm")

    def _execute_llm_generation(
        self,
        prompt: str,
        used_evidence: bool,
        evidence_ids: List[str],
        method_name: str
    ) -> Dict[str, Any]:
        from src.rate_limiter import rate_limited_api_call

        def _do_llm():
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=300
                )
            )
            reply_text = response.text.strip()
            if reply_text.startswith('"') and reply_text.endswith('"'):
                reply_text = reply_text[1:-1].strip()

            return {
                "reply": reply_text,
                "used_evidence": used_evidence,
                "evidence_ids": evidence_ids,
                "confidence": 0.90,
                "method": method_name
            }

        def _fallback():
            return {
                "reply": "[LLM API Quota Exhausted]",
                "used_evidence": used_evidence,
                "evidence_ids": evidence_ids,
                "confidence": 0.0,
                "method": f"{method_name}_fallback"
            }

        return rate_limited_api_call(call_fn=_do_llm, max_retries=2, fallback_fn=_fallback)

    def generate_reply(
        self,
        customer_text: str,
        intent: str = "other",
        escalation_decision: str = "auto_handle",
        escalation_reason: str = "",
        method: str = "rag_llm"
    ) -> Dict[str, Any]:
        """Unified reply generation interface."""
        if method == "canned":
            return self.generate_canned(customer_text)
        elif method == "template":
            return self.generate_template(customer_text, intent, escalation_decision, escalation_reason)
        elif method == "llm_no_rag":
            return self.generate_llm_no_rag(customer_text, intent, escalation_decision, escalation_reason)
        elif method == "rag_llm":
            return self.generate_rag_llm(customer_text, intent, escalation_decision, escalation_reason)
        else:
            return self.generate_template(customer_text, intent, escalation_decision, escalation_reason)

    def _build_generation_prompt(
        self,
        customer_text: str,
        intent: str,
        retrieval_context: str,
        escalation_decision: str,
        escalation_reason: str
    ) -> str:
        escalation_guidance = ""
        if escalation_decision == "escalate":
            escalation_guidance = f"""
SPECIAL ESCALATION PROTOCOL (HUMAN ESCALATION MANDATED):
Reason: {escalation_reason}
- If physical safety (swelling, fire, smoke, sparks): Instruct user to immediately stop using and charging the device, disconnect power, and visit an Apple Store or Apple Support safety specialist.
- If account security/lockout: Guide user to https://iforgot.apple.com to secure credentials, or invite them to DM for specialist review.
- If legal threat, manager request, or financial dispute: Invite user to Direct Message so a Senior Specialist can review their case.
DO NOT provide standard routine reboot/update troubleshooting for escalated safety/security issues.
"""

        return f"""You are a customer support specialist for Apple Support on Twitter (@AppleSupport).
Draft a concise, empathetic, and fully grounded reply to the customer's tweet.

STRICT OPERATIONAL RULES:
1. Conciseness: Fit within Twitter constraints (under 280 characters).
2. Brand Voice: Warm, empathetic, professional, solution-oriented ("We'd like to help", "Let's work together").
3. Grounding: Rely strictly on verified Apple protocols. NEVER invent company policy, refund guarantees, pricing, or unverified timelines.
4. Portal Links: Point to official portals where appropriate:
   - Password/Account: https://iforgot.apple.com
   - Refund/Billing: https://reportaproblem.apple.com
   - Coverage/Warranty: https://checkcoverage.apple.com
   - Direct Message: https://twitter.com/messages/compose?recipient_id=AppleSupport
5. Inadequate Evidence: If historical evidence is insufficient, ask a clarifying question rather than guessing.
{escalation_guidance}
HISTORICAL RESOLUTION EVIDENCE:
{retrieval_context}

CUSTOMER INTENT: {intent}
ESCALATION STATUS: {escalation_decision}

CUSTOMER TWEET:
"{customer_text}"

Return ONLY the plain-text reply to be sent (no quotes, no markdown, no preamble):"""
