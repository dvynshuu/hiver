import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ESCALATION_CRITICAL_INTENTS,
    ESCALATION_KEYWORDS,
    GEMINI_API_KEY,
    AGENT_MODEL_NAME
)

logger = logging.getLogger(__name__)

# Specific high-urgency keywords and patterns
SAFETY_KEYWORDS = [
    "exploded", "explod", "caught fire", "catch fire", "fire", "smoke", "smoking",
    "swelling", "swollen", "bulging", "burned", "burning", "sparks", "sparking",
    "electric shock", "shocked me", "melted", "melting"
]
LEGAL_KEYWORDS = [
    "lawsuit", "attorney", "lawyer", "sue", "suing", "court", "bbb",
    "better business bureau", "legal action", "class action", "consumer rights"
]
HUMAN_KEYWORDS = [
    "real person", "human agent", "speak with someone", "talk to someone",
    "speak to a human", "talk to a human", "real human", "talk to a person",
    "speak to a person", "transfer to a human", "need a human",
    "supervisor", "representative", "manager",
    "transfer me to", "connect me with"
]
SECURITY_KEYWORDS = [
    "hacked", "stolen", "breach", "compromised", "identity theft", "unauthorized",
    "fraud", "blackmail", "spyware", "sim swap"
]


class EscalationEngine:
    """
    Decides whether an incoming customer message should be auto-handled
    by the AI agent or escalated to a human support representative.
    
    Provides explicit reasoning for every decision to establish human trust.
    Uses a hybrid architecture:
    1. Deterministic safety & compliance guardrails (safety hazard, legal threat, human request)
    2. Intent-based business policy (account security, high-risk billing)
    3. Confidence & sentiment reasoning
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
                logger.warning(f"Could not initialize Gemini client for escalation: {e}")

    def decide(
        self,
        customer_text: str,
        intent: str = "other",
        intent_confidence: float = 1.0,
        thread_turn_count: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate customer inquiry and decide whether to auto-handle or escalate.
        Returns:
            decision: 'auto_handle' | 'escalate'
            reason: detailed explanation of the decision
            urgency: 'low' | 'medium' | 'high' | 'critical'
            trigger: the specific rule or policy that triggered the decision
            confidence: confidence score for the decision
        """
        text_lower = customer_text.lower()

        # -------------------------------------------------------------
        # 1. IMMEDIATE SAFETY HAZARD ESCALATIONS (CRITICAL)
        # -------------------------------------------------------------
        for kw in SAFETY_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                return {
                    "decision": "escalate",
                    "reason": f"Physical safety concern detected ('{kw}'). Requires immediate human AppleCare safety protocol.",
                    "urgency": "critical",
                    "trigger": "safety_hazard",
                    "confidence": 0.99
                }

        # -------------------------------------------------------------
        # 2. LEGAL / REGULATORY THREATS (HIGH)
        # -------------------------------------------------------------
        for kw in LEGAL_KEYWORDS:
            if re.search(rf"\b{kw}\b", text_lower):
                return {
                    "decision": "escalate",
                    "reason": f"Legal threat or regulatory mention detected ('{kw}'). Must be routed to specialist legal support.",
                    "urgency": "high",
                    "trigger": "legal_threat",
                    "confidence": 0.98
                }

        # -------------------------------------------------------------
        # 3. EXPLICIT HUMAN AGENT REQUEST (MEDIUM/HIGH)
        # -------------------------------------------------------------
        for kw in HUMAN_KEYWORDS:
            if kw in text_lower:
                return {
                    "decision": "escalate",
                    "reason": f"Customer explicitly requested a human specialist or manager ('{kw}').",
                    "urgency": "medium",
                    "trigger": "human_requested",
                    "confidence": 0.95
                }

        # -------------------------------------------------------------
        # 4. ACCOUNT SECURITY & COMPROMISE (HIGH)
        # -------------------------------------------------------------
        if intent == "account_security":
            return {
                "decision": "escalate",
                "reason": "Account security and Apple ID credentials involve confidential PII and 2FA verification requiring authenticated human support.",
                "urgency": "high",
                "trigger": "critical_intent_security",
                "confidence": 0.94
            }

        for kw in SECURITY_KEYWORDS:
            if kw in text_lower:
                return {
                    "decision": "escalate",
                    "reason": f"Security compromise signal detected ('{kw}'). Requires human agent verification.",
                    "urgency": "high",
                    "trigger": "security_compromise",
                    "confidence": 0.93
                }

        # -------------------------------------------------------------
        # 5. DISPUTED BILLING / REFUND ESCALATIONS (MEDIUM)
        # -------------------------------------------------------------
        if intent == "billing_purchase":
            if any(term in text_lower for term in ["unauthorized", "stolen", "fraud", "dispute", "scam", "twice", "double charged"]):
                return {
                    "decision": "escalate",
                    "reason": "Financial transaction dispute / unauthorized charge requires transactional human review.",
                    "urgency": "medium",
                    "trigger": "financial_dispute",
                    "confidence": 0.91
                }

        # -------------------------------------------------------------
        # 6. MULTI-TURN THREAD FRUSTRATION (>3 TURNS)
        # -------------------------------------------------------------
        if thread_turn_count >= 3:
            return {
                "decision": "escalate",
                "reason": f"Conversation has exceeded {thread_turn_count} turns without resolution; escalating to human agent to prevent customer fatigue.",
                "urgency": "medium",
                "trigger": "thread_fatigue",
                "confidence": 0.88
            }

        # -------------------------------------------------------------
        # 7. LOW INTENT CLASSIFICATION CONFIDENCE (<0.55)
        # -------------------------------------------------------------
        if intent_confidence < 0.55 and intent != "other":
            return {
                "decision": "escalate",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f}). Escalating to human to prevent incorrect automated advice.",
                "urgency": "low",
                "trigger": "low_confidence",
                "confidence": 0.80
            }

        # -------------------------------------------------------------
        # 8. GENERAL FEEDBACK / PRAISE (AUTO-HANDLE)
        # -------------------------------------------------------------
        if intent == "general_feedback":
            if any(praise in text_lower for praise in ["thank", "great", "love", "best", "awesome"]):
                return {
                    "decision": "auto_handle",
                    "reason": "Customer expressed positive feedback or appreciation; suitable for polite automated acknowledgment.",
                    "urgency": "low",
                    "trigger": "praise_feedback",
                    "confidence": 0.95
                }

        # -------------------------------------------------------------
        # 9. STANDARD TROUBLESHOOTING & PRODUCT INQUIRIES (AUTO-HANDLE)
        # -------------------------------------------------------------
        if intent in {"device_issue", "software_bug", "connectivity", "product_inquiry", "other"}:
            return {
                "decision": "auto_handle",
                "reason": f"Standard {intent.replace('_', ' ')} inquiry with established troubleshooting procedures and historical resolution precedents.",
                "urgency": "low",
                "trigger": "standard_troubleshooting",
                "confidence": 0.89
            }

        # Default fallback
        return {
            "decision": "auto_handle",
            "reason": "Inquiry matches standard support domain without escalation risk factors.",
            "urgency": "low",
            "trigger": "default_policy",
            "confidence": 0.75
        }
