import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.escalation_engine import EscalationEngine

class TestEscalationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = EscalationEngine()

    def test_safety_hazard_escalation(self):
        """Verify linguistic variants of battery swelling and thermal hazards escalate with critical urgency."""
        queries = [
            "My iPhone battery is swelling up and pushing the screen out",
            "The back of my phone is puffy and feels unusually thick",
            "Screen is lifting away from the frame with a chemical smell",
            "My charger sparked and started smoking while plugged in",
            "The phone got burning hot on my desk and scorched the surface",
            "my battery is swelling",
            "the phone is getting hot",
            "screen is lifting",
            "battery smells weird"
        ]
        for q in queries:
            res = self.engine.decide(q)
            self.assertEqual(res["decision"], "escalate", f"Failed to escalate safety hazard: {q}")
            self.assertEqual(res["urgency"], "critical")
            self.assertEqual(res["trigger"], "safety_hazard")

    def test_legal_threat_escalation(self):
        """Verify legal actions and regulatory complaints escalate with high urgency."""
        queries = [
            "I have retained an attorney and am filing a lawsuit against Apple",
            "Filing a formal deceptive trade practices complaint with the Better Business Bureau",
            "Our legal counsel has instructed us to preserve records for court"
        ]
        for q in queries:
            res = self.engine.decide(q)
            self.assertEqual(res["decision"], "escalate")
            self.assertEqual(res["urgency"], "high")
            self.assertEqual(res["trigger"], "legal_threat")

    def test_human_agent_request_escalation(self):
        """Verify explicit demands for a human, supervisor, or manager escalate."""
        queries = [
            "I refuse to talk to an automated bot, transfer me to a real person",
            "Connect me with a manager or supervisor immediately",
            "Transfer me to a live support representative"
        ]
        for q in queries:
            res = self.engine.decide(q)
            self.assertEqual(res["decision"], "escalate")
            self.assertEqual(res["trigger"], "human_requested")

    def test_multi_turn_fatigue_escalation(self):
        """Verify inquiries exceeding turn limit escalate to prevent bot fatigue."""
        res = self.engine.decide("Still having this issue after trying your steps", thread_turn_count=3)
        self.assertEqual(res["decision"], "escalate")
        self.assertEqual(res["trigger"], "thread_fatigue")

    def test_routine_troubleshooting_autohandle(self):
        """Verify routine technical inquiries are marked auto_handle."""
        queries = [
            "How do I update my iPhone 12 to iOS 17?",
            "What is the difference between Apple Pencil 1 and 2?",
            "Thanks for the help today, appreciate the quick reply!"
        ]
        for q in queries:
            res = self.engine.decide(q, intent="software_bug")
            self.assertEqual(res["decision"], "auto_handle")
            self.assertEqual(res["urgency"], "low")

if __name__ == "__main__":
    unittest.main()
