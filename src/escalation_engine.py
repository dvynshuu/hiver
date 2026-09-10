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

# Expanded lemma-aware regular expressions for safety & hazard detection
SAFETY_PATTERNS = [
    r"\b(swell|swells|swelling|swollen|bulg|bulging|bulged)\b",
    r"\b(puffy|puffing|puff|puffed)\b",
    r"\b(expand|expands|expanded|expanding|widening|thicker)\b",
    r"\b(screen.*lifting|glass.*lifted|screen.*detached|frame.*separat)\b",
    r"\b(smoke|smoking|smoked|fumes|smell.*chemical|strange.*smell)\b",
    r"\b(fire|caught fire|catch fire|flames|flame)\b",
    r"\b(burn|burned|burning|burnt|scorch|scorched|scorching)\b",
    r"\b(explod|exploding|exploded|explosion|popped.*smoke|blast)\b",
    r"\b(spark|sparks|sparking|sparked|electric shock|shocked me|zapped)\b",
    r"\b(melt|melted|melting)\b",
    r"\b(burning hot|scorching hot|blistering hot)\b"
]

LEGAL_PATTERNS = [
    r"\b(lawsuit|attorney|lawyer|sue|suing|sued|court)\b",
    r"\b(bbb|better business bureau|ftc|federal trade commission)\b",
    r"\b(legal action|class action|legal counsel|consumer rights|consumer protection)\b"
]

HUMAN_PATTERNS = [
    r"\b(real person|human agent|speak with someone|talk to someone)\b",
    r"\b(speak to a human|talk to a human|real human|talk to a person)\b",
    r"\b(speak to a person|transfer to a human|need a human)\b",
    r"\b(supervisor|representative|manager|senior advisor|live agent)\b",
    r"\b(transfer me|connect me with|not a bot|refuse.*bot)\b"
]

SECURITY_PATTERNS = [
    r"\b(hacked|stolen|breach|compromised|identity theft|unauthorized)\b",
    r"\b(fraud|blackmail|spyware|sim swap|sim swapped)\b",
    r"\b(locked out|account locked|2fa.*code|verification code.*scam)\b"
]

FINANCIAL_PATTERNS = [
    # Explicit unauthorized / unrecognized
    r"\b(unauthorized|unapproved|unrecognized|unknown|unexpected)\s+(?:[\w\./-]+\s+)*(charge|payment|transaction|purchase|fee|debit|bill|order|deduction)\b",
    r"\b(charge|payment|transaction|purchase)\s+(wasn'?t mine|is not mine|was not mine)\b",
    r"\b(don'?t recognize|do not recognize|didn'?t authorize|did not authorize)\s+(this|the|that|my)?\s*(charge|payment|transaction|purchase|order)?\b",
    r"\b(apple\.com/bill|itunes\.com/bill)\b",
    
    # Duplicate / multiple charges
    r"\b(double|duplicate|repeat|extra)\s+(charge|payment|transaction|billing|billed)\b",
    r"\b(charged|billed|paid)\s+(twice|two times|double|again|multiple times|\d+\s*times)\b",
    r"\bwhy\s+(was i|am i being|did you)\s+(charge|charged)\b",
    r"\bwithout\s+(?:my\s+)?(authorization|permission|consent|approval)\b",
    r"\b(charge|charged|debit|debited)\s+(?:my\s+)?(?:card|debit card|credit card|account)\s+without\b",
    
    # Fraud / stolen card
    r"\b(someone|somebody|thief)\s+used\s+my\s+(card|account|credit card|debit card|apple pay|apple card)\b",
    r"\b(card|credit card|debit card|apple card)\s+(was\s+)?stolen\b",
    r"\b(stolen\s+card|card\s+theft|compromised\s+card)\b",
    r"\b(fraud|fraudulent)\s+(charge|transaction|purchase|activity|payment|billing)\b",
    r"\b(credit card|debit card|bank)\s+fraud\b",
    
    # Disputes & failed refunds
    r"\b(dispute|disputing|disputed)\s+(this|the|a)?\s*(charge|payment|transaction|bill)\b",
    r"\b(refund\s+(declined|denied|rejected|refused)|denied\s+(my\s+)?refund)\b",
    r"\b(overcharged|overcharge|overcharging)\b",
    r"\b(scam|scammed|scammer)\s+(charge|purchase|payment|transaction)\b"
]

class EscalationEngine:
    """
    Decides whether an incoming customer inquiry should be auto-handled
    by the AI agent or escalated to a human support specialist.

    Enforces deterministic safety bounds:
    1. Critical Physical Hazards (swelling, fire, smoke, sparks, explosions)
    2. Legal & Regulatory Threats (attorney, lawsuit, FTC, BBB)
    3. Account Security & Takeover (compromised credentials, SIM swap)
    4. Financial Disputes & Fraud (unauthorized charges, double billing)
    5. Explicit Human / Supervisor Demands
    6. Multi-Turn Thread Fatigue (>= 3 turns without resolution)
    7. High Uncertainty (low intent classification confidence)
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = AGENT_MODEL_NAME):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name

    def decide(
        self,
        customer_text: str,
        intent: str = "other",
        intent_confidence: float = 1.0,
        thread_turn_count: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate inquiry against deterministic safety policies.
        Returns:
            decision: 'auto_handle' | 'escalate'
            reason: human-readable explanation of why this decision was made
            urgency: 'low' | 'medium' | 'high' | 'critical'
            trigger: policy rule identifier
            confidence: decision confidence
        """
        text_lower = customer_text.lower()

        # -------------------------------------------------------------
        # 1. CRITICAL PHYSICAL & THERMAL SAFETY HAZARDS (CRITICAL)
        # -------------------------------------------------------------
        for pat in SAFETY_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                matched_kw = match.group(0)
                return {
                    "decision": "escalate",
                    "reason": f"Physical safety hazard detected ('{matched_kw}'). Requires immediate human AppleCare safety protocol.",
                    "urgency": "critical",
                    "trigger": "safety_hazard",
                    "confidence": 0.99
                }

        # -------------------------------------------------------------
        # 2. LEGAL THREATS & REGULATORY COMPLAINTS (HIGH)
        # -------------------------------------------------------------
        for pat in LEGAL_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                matched_kw = match.group(0)
                return {
                    "decision": "escalate",
                    "reason": f"Legal threat or regulatory mention detected ('{matched_kw}'). Routed to Senior Customer Relations / Legal.",
                    "urgency": "high",
                    "trigger": "legal_threat",
                    "confidence": 0.98
                }

        # -------------------------------------------------------------
        # 3. EXPLICIT HUMAN AGENT / MANAGER REQUEST (MEDIUM/HIGH)
        # -------------------------------------------------------------
        for pat in HUMAN_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                matched_kw = match.group(0)
                return {
                    "decision": "escalate",
                    "reason": f"Customer explicitly requested a human specialist or manager ('{matched_kw}').",
                    "urgency": "medium",
                    "trigger": "human_requested",
                    "confidence": 0.96
                }

        # -------------------------------------------------------------
        # 4. ACCOUNT SECURITY & IDENTITY COMPROMISE (HIGH)
        # -------------------------------------------------------------
        for pat in SECURITY_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                matched_kw = match.group(0)
                return {
                    "decision": "escalate",
                    "reason": f"Security breach signal detected ('{matched_kw}'). Requires authenticated human verification.",
                    "urgency": "high",
                    "trigger": "security_compromise",
                    "confidence": 0.94
                }

        if intent == "account_security":
            return {
                "decision": "escalate",
                "reason": "Account security and Apple ID credentials involve confidential PII and 2FA verification requiring authenticated support.",
                "urgency": "high",
                "trigger": "critical_intent_security",
                "confidence": 0.93
            }

        # -------------------------------------------------------------
        # 5. FINANCIAL DISPUTES & TRANSACTION FRAUD (MEDIUM/HIGH)
        # -------------------------------------------------------------
        for pat in FINANCIAL_PATTERNS:
            match = re.search(pat, text_lower)
            if match:
                matched_kw = match.group(0)
                return {
                    "decision": "escalate",
                    "reason": f"Financial transaction dispute detected ('{matched_kw}'). Routed to human billing specialist.",
                    "urgency": "high",
                    "trigger": "financial_dispute",
                    "confidence": 0.92
                }

        # Policy rule: If classified as billing_purchase with dispute/fraud indicators
        if intent == "billing_purchase":
            dispute_terms = [
                "dispute", "fraud", "unauthorized", "duplicate", "twice", "stolen", 
                "refund", "declined", "denied", "charged", "overcharged", "not mine", 
                "recognize", "scam", "wrong amount", "unknown charge", "double"
            ]
            if any(term in text_lower for term in dispute_terms):
                return {
                    "decision": "escalate",
                    "reason": "Billing inquiry involves financial dispute or transaction complaint requiring specialist review.",
                    "urgency": "high",
                    "trigger": "financial_dispute",
                    "confidence": 0.94
                }

        # -------------------------------------------------------------
        # 6. MULTI-TURN THREAD FATIGUE (>= 3 TURNS)
        # -------------------------------------------------------------
        if thread_turn_count >= 3:
            return {
                "decision": "escalate",
                "reason": f"Conversation exceeded {thread_turn_count} turns without resolution. Escalate to prevent bot loop fatigue.",
                "urgency": "medium",
                "trigger": "thread_fatigue",
                "confidence": 0.88
            }

        # -------------------------------------------------------------
        # 7. LOW INTENT CONFIDENCE (< 0.55)
        # -------------------------------------------------------------
        if intent_confidence < 0.55 and intent != "other":
            return {
                "decision": "escalate",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f}). Escalate to prevent automated error.",
                "urgency": "low",
                "trigger": "low_confidence",
                "confidence": 0.80
            }

        # -------------------------------------------------------------
        # 8. STANDARD AUTO-HANDLE PROTOCOL
        # -------------------------------------------------------------
        return {
            "decision": "auto_handle",
            "reason": f"Inquiry matches standard support domain ({intent.replace('_', ' ')}) without high-risk escalation triggers.",
            "urgency": "low",
            "trigger": "standard_support",
            "confidence": 0.89
        }

    def evaluate_adversarial_suite(
        self,
        adversarial_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate escalation recall on safety, security, financial, and legal adversarial cases.
        """
        categories = {
            "safety_hazard": {"total": 0, "caught": 0},
            "security": {"total": 0, "caught": 0},
            "financial": {"total": 0, "caught": 0},
            "legal": {"total": 0, "caught": 0},
            "human_request": {"total": 0, "caught": 0}
        }

        total_adv = 0
        total_caught = 0

        for item in adversarial_items:
            text = item.get("customer_message", item.get("customer_text", ""))
            intent = item.get("intent", item.get("ground_truth_intent", "other"))
            trigger = item.get("escalation_trigger", "")

            dec = self.decide(text, intent=intent)
            is_caught = (dec["decision"] == "escalate")

            total_adv += 1
            if is_caught:
                total_caught += 1

            # Map to category
            t_low = trigger.lower()
            if "safety" in t_low or "swell" in t_low:
                cat = "safety_hazard"
            elif "security" in t_low or "takeover" in t_low or "fraud" in t_low or "sim" in t_low:
                cat = "security"
            elif "billing" in t_low or "financial" in t_low or "charge" in t_low:
                cat = "financial"
            elif "legal" in t_low or "regulatory" in t_low:
                cat = "legal"
            elif "human" in t_low or "manager" in t_low:
                cat = "human_request"
            else:
                cat = "safety_hazard"

            categories[cat]["total"] += 1
            if is_caught:
                categories[cat]["caught"] += 1

        results = {
            "overall_adversarial_recall": round(total_caught / total_adv, 3) if total_adv else 1.0,
            "overall_critical_risk_recall": round(total_caught / total_adv, 3) if total_adv else 1.0,
            "physical_safety_recall": round(categories["safety_hazard"]["caught"] / categories["safety_hazard"]["total"], 3) if categories["safety_hazard"]["total"] else 1.0,
            "security_recall": round(categories["security"]["caught"] / categories["security"]["total"], 3) if categories["security"]["total"] else 1.0,
            "financial_recall": round(categories["financial"]["caught"] / categories["financial"]["total"], 3) if categories["financial"]["total"] else 1.0,
            "legal_recall": round(categories["legal"]["caught"] / categories["legal"]["total"], 3) if categories["legal"]["total"] else 1.0,
            "human_request_recall": round(categories["human_request"]["caught"] / categories["human_request"]["total"], 3) if categories["human_request"]["total"] else 1.0,
            "total_adversarial_tested": total_adv,
            "total_adversarial_caught": total_caught,
            "categories": {}
        }

        for cat, data in categories.items():
            if data["total"] > 0:
                results["categories"][cat] = {
                    "recall": round(data["caught"] / data["total"], 3),
                    "caught": data["caught"],
                    "total": data["total"]
                }

        return results
