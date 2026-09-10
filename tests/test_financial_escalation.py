import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation_engine import EscalationEngine

class TestFinancialEscalation(unittest.TestCase):
    def setUp(self):
        self.engine = EscalationEngine()

    def test_financial_dispute_semantic_paraphrases(self):
        """Verify various real-world phrasing of unauthorized charges and disputes trigger escalation."""
        paraphrases = [
            "I don't recognize this payment",
            "Why was I charged twice for the same app?",
            "This transaction wasn't mine",
            "Someone used my card on iTunes without my permission",
            "I want to dispute this charge on my credit card statement",
            "There is an unknown apple.com/bill charge on my account",
            "I noticed an unrecognized recurring subscription fee",
            "Fraudulent transaction appeared on my Apple Pay",
            "Why did you charge my debit card without authorization?"
        ]
        
        for text in paraphrases:
            res = self.engine.decide(text)
            self.assertEqual(
                res["decision"], "escalate",
                f"Failed to escalate financial dispute query: '{text}'"
            )
            self.assertIn(res["urgency"], ["high", "critical"])
            self.assertEqual(res["trigger"], "financial_dispute")

    def test_policy_escalation_on_billing_intent_with_dispute(self):
        """Verify policy-level escalation triggers when intent is billing_purchase and dispute indicators exist."""
        text = "I need to dispute an accidental in-app purchase my kid made"
        res = self.engine.decide(text, intent="billing_purchase")
        self.assertEqual(res["decision"], "escalate")

if __name__ == "__main__":
    unittest.main()
