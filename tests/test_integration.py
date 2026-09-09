import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent_classifier import IntentClassifier
from src.retrieval import HistoricalRetrievalEngine
from src.escalation_engine import EscalationEngine
from src.reply_generator import ReplyGenerator

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.intent_classifier = IntentClassifier()
        self.retrieval_engine = HistoricalRetrievalEngine.load()
        self.escalation_engine = EscalationEngine()
        self.reply_generator = ReplyGenerator(retrieval_engine=self.retrieval_engine)

    def test_end_to_end_standard_inquiry(self):
        """Verify full agent pipeline execution on routine troubleshooting inquiry without gold labels."""
        query = "How do I update to iOS 17 on my iPhone?"

        # Stage 1: Intent
        intent_res = self.intent_classifier.classify(query, method="learned")
        self.assertIn("intent", intent_res)
        pred_intent = intent_res["intent"]

        # Stage 2: Escalation
        esc_res = self.escalation_engine.decide(query, intent=pred_intent, intent_confidence=intent_res["confidence"])
        self.assertEqual(esc_res["decision"], "auto_handle")

        # Stage 3: Retrieval
        retrieved = self.retrieval_engine.retrieve(query, top_k=2)
        self.assertIsInstance(retrieved, list)

        # Stage 4: Reply Generation
        reply_res = self.reply_generator.generate_reply(
            customer_text=query,
            intent=pred_intent,
            escalation_decision=esc_res["decision"],
            escalation_reason=esc_res["reason"],
            method="template"
        )
        self.assertIn("reply", reply_res)
        self.assertGreater(len(reply_res["reply"]), 10)
        self.assertLessEqual(len(reply_res["reply"]), 280)

    def test_end_to_end_safety_escalation(self):
        """Verify full agent pipeline execution on hazardous inquiry triggers escalation instructions."""
        hazard_query = "My battery is swelling up and pushing the screen out of the phone"

        # Stage 1: Intent
        intent_res = self.intent_classifier.classify(hazard_query, method="learned")

        # Stage 2: Escalation (Deterministic guardrail catches hazard regardless of intent)
        esc_res = self.escalation_engine.decide(hazard_query, intent=intent_res["intent"], intent_confidence=intent_res["confidence"])
        self.assertEqual(esc_res["decision"], "escalate")
        self.assertEqual(esc_res["urgency"], "critical")

        # Stage 3: Reply Generation (Pivots to safety instructions, NOT routine reboot)
        reply_res = self.reply_generator.generate_reply(
            customer_text=hazard_query,
            intent=intent_res["intent"],
            escalation_decision=esc_res["decision"],
            escalation_reason=esc_res["reason"],
            method="template"
        )
        self.assertIn("safety", reply_res["reply"].lower())
        self.assertNotIn("restart your device", reply_res["reply"].lower())

if __name__ == "__main__":
    unittest.main()
