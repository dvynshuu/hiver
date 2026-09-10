"""
Compile the verified, human-annotated Golden Evaluation Set for @AppleSupport.
Strictly compiles manual annotations from data/golden/manual_annotations.jsonl,
enforcing explicit provenance, zero leakage, and intent coverage.

DOES NOT generate heuristic labels as ground truth.
Workflow:
candidate generation -> manual annotation -> golden dataset -> evaluation.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import (
    DATA_DIR,
    RETRIEVAL_CORPUS_JSONL_PATH,
    GOLDEN_EVAL_PATH,
    INTENT_NAMES
)
from src.data_pipeline import check_evaluation_leakage, format_leakage_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DIR = DATA_DIR / "golden"
MANUAL_ANNOTATIONS_PATH = GOLDEN_DIR / "manual_annotations.jsonl"
CANDIDATES_POOL_PATH = GOLDEN_DIR / "candidates_pool.jsonl"

ADVERSARIAL_CASES = [
    {
        "text": "My iPhone battery started swelling and the screen is lifting away from the metal frame! What do I do?!",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Physical safety hazard: battery swelling and frame separation indicates thermal runaway risk.",
        "reply": "Your safety is our top priority. Please stop using and charging the device immediately, keep it in a safe area, and contact Apple Support directly or visit an Apple Store for priority safety evaluation."
    },
    {
        "text": "The back glass of my phone looks puffy and feels unusually thick like something inside is expanding.",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Physical safety hazard: puffy enclosure indicates lithium-ion pouch expansion.",
        "reply": "Please stop charging and powering the device immediately. This indicates possible battery expansion. Bring it into an Apple Authorized Service Provider or DM us so we can guide you to immediate safety inspection."
    },
    {
        "text": "Screen is lifting up on the left side and there is a strange chemical sweet smell coming from the seam.",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Physical safety hazard: chemical odor paired with screen displacement indicates battery cell puncture/off-gassing.",
        "reply": "For your safety, immediately power down the device and avoid inhaling any vapors. Do not connect to power. Contact Apple Support directly or visit an Apple Store right away."
    },
    {
        "text": "The lightning cable that came with my iPhone started smoking and melted the connector while I was asleep!",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Fire hazard: melted charging cable and smoke emission requires immediate safety incident routing.",
        "reply": "We take safety reports very seriously. Please ensure the charger is completely unplugged from the wall. Send us a Direct Message with your details so our safety engineering team can inspect this immediately: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "My iPad got burning hot to the touch on the desk, sparks flew from the port, and it scorched my wooden desk.",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Fire hazard: port sparking and thermal scorch damage requires priority AppleCare safety escalation.",
        "reply": "Safety is our absolute priority. Disconnect the device from power and keep it isolated on a non-flammable surface. DM us your contact info right away so a Senior Specialist can reach out: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "Device battery popped with a loud pop sound and black smoke filled my bedroom!",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "safety_hazard",
        "reason": "Critical physical hazard: battery combustion/smoke discharge requires immediate specialist handling.",
        "reply": "Ensure everyone is in a well-ventilated area away from the device. Please DM us immediately with your phone number and location so our specialized safety team can assist: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "Someone in another country hacked my Apple ID, changed my recovery email, and locked me out of iCloud!",
        "intent": "account_security",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "account_takeover",
        "reason": "Account takeover: credential alteration and unauthorized foreign access requires human security team triage.",
        "reply": "We take account security very seriously. Please head to https://iforgot.apple.com right away to attempt recovery, and DM us your details so our security team can assist you directly: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "My cellular carrier informed me of an unauthorized SIM swap and now someone is resetting my Apple ID password!",
        "intent": "account_security",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "sim_swap_takeover",
        "reason": "SIM swap attack: active interception of 2FA codes demands urgent account freeze protocol.",
        "reply": "This requires immediate action. Visit https://iforgot.apple.com to lock down your credentials and join us in a Direct Message right now so our account security specialists can assist: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "A caller claiming to be Apple Support scammed my elderly father into reading out his 6-digit verification code.",
        "intent": "account_security",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "fraud_social_engineering",
        "reason": "Social engineering / 2FA credential compromise requiring fraud specialist intervention.",
        "reply": "Apple will never call asking for your verification code. Please change the Apple ID password immediately at https://appleid.apple.com and send us a DM so we can secure the account: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "There is a $499 unauthorized charge on my Apple Card for electronics I never ordered in another state.",
        "intent": "billing_purchase",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "fraud_unauthorized_charge",
        "reason": "Fraudulent billing transaction on Apple Card requires human dispute specialist review.",
        "reply": "We want to help resolve this unauthorized charge. Open the Wallet app, tap your Apple Card, select the charge, and choose 'Report an Issue', or send us a DM so we can connect you with Apple Card support."
    },
    {
        "text": "I have been double-billed for Apple Music for 8 consecutive months and your automated refund system declined me!",
        "intent": "billing_purchase",
        "escalate": True,
        "urgency": "medium",
        "difficulty": "hard",
        "trigger": "recurring_billing_dispute",
        "reason": "Recurring multi-month billing error with prior automated system failure requires human escalation.",
        "reply": "We understand how frustrating recurring charges are. Since the automated system declined it, please DM us your order ID and Apple ID email so a billing specialist can review your case: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "My credit card was charged 12 separate times in 10 minutes for in-app purchases I didn't authorize.",
        "intent": "billing_purchase",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "rapid_fraudulent_charges",
        "reason": "High-frequency in-app purchase fraud requires immediate billing specialist review and card block.",
        "reply": "We want to protect your payment method immediately. Report the charges at https://reportaproblem.apple.com and DM us so we can review the account activity with you directly."
    },
    {
        "text": "Your repair refusal violates consumer protection laws. My attorney has drafted a lawsuit and will serve Apple this week.",
        "intent": "general_feedback",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "legal_threat",
        "reason": "Litigation threat and attorney involvement requires routing to Apple legal escalation procedures.",
        "reply": "We take these concerns seriously. Please send us a Direct Message with your case number and contact information so our Senior Customer Relations team can review your file: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "I am filing a formal deceptive trade practices complaint with the Better Business Bureau and FTC regarding this charge.",
        "intent": "billing_purchase",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "regulatory_threat",
        "reason": "Regulatory complaint filing (FTC / BBB) mandates human customer relations handling.",
        "reply": "We want the opportunity to make this right. Please join us in a Direct Message with your billing details and case number so a Senior Specialist can review this with you: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "Our legal counsel has formally instructed us to preserve all records regarding this battery explosion incident.",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "legal_safety_combination",
        "reason": "Legal counsel notice combined with physical safety incident mandates executive safety relations.",
        "reply": "Thank you for notifying us. Please DM us your direct phone number, case number, and counsel contact info so our Senior Executive Relations team can reach out immediately: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "I refuse to speak to an automated bot. Transfer me to a live human representative or supervisor right now.",
        "intent": "general_feedback",
        "escalate": True,
        "urgency": "medium",
        "difficulty": "medium",
        "trigger": "human_agent_requested",
        "reason": "Customer explicitly rejected automated assistance and requested human representative/supervisor.",
        "reply": "We completely understand and are happy to connect you with our team. Please tap this link to send us a Direct Message and a support specialist will assist you: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "Connect me with a manager or senior advisor. Three agents have given me contradictory answers today.",
        "intent": "general_feedback",
        "escalate": True,
        "urgency": "medium",
        "difficulty": "medium",
        "trigger": "manager_requested",
        "reason": "Customer requested manager/senior advisor following repeated contradictory service experiences.",
        "reply": "We are sorry for the inconsistent information you received. Let's get a Senior Advisor to review this with you. Please DM us your case number: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "Is there a real human being monitoring this Twitter account? I need to talk to a person.",
        "intent": "other",
        "escalate": True,
        "urgency": "medium",
        "difficulty": "easy",
        "trigger": "human_requested",
        "reason": "Explicit inquiry requesting human communication channel.",
        "reply": "Yes! Our support team is here and ready to help. Select the link below to meet us in a Direct Message and we'll take care of you: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    },
    {
        "text": "My phone is getting noticeably wider and won't fit into its case anymore. Is this a common issue with iPhone?",
        "intent": "device_issue",
        "escalate": True,
        "urgency": "critical",
        "difficulty": "hard",
        "trigger": "subtle_battery_expansion",
        "reason": "Subtle indicator of battery swelling (device widening and case misfit). Poses fire hazard.",
        "reply": "A device expanding or not fitting its case can indicate a swollen battery. For your safety, stop charging it immediately and visit an Apple Store or DM us so we can arrange priority inspection."
    },
    {
        "text": "An unauthorized third party registered an Apple ID using our university department's administrative email domain.",
        "intent": "account_security",
        "escalate": True,
        "urgency": "high",
        "difficulty": "hard",
        "trigger": "domain_identity_conflict",
        "reason": "Enterprise domain identity conflict and unauthorized Apple ID registration requires administrative verification.",
        "reply": "We can help route this to our Apple Deployment and Security specialists. Please DM us your organization details and the affected domain so we can initiate domain verification: https://twitter.com/messages/compose?recipient_id=AppleSupport"
    }
]

def classify_text_heuristically(text: str) -> str:
    """Classify text into 8 intents using deterministic domain rules."""
    t = text.lower()
    if any(k in t for k in ["battery", "charge", "screen", "display", "speaker", "audio", "mic", "hardware", "camera", "button", "volume", "overheating", "swell", "hot", "vibrat", "headphone"]):
        if any(b in t for b in ["refund", "double charge", "card declined", "invoice", "receipt", "subscription", "billed"]):
            return "billing_purchase"
        return "device_issue"
    if any(k in t for k in ["apple id", "icloud", "password", "passcode", "locked", "verification code", "2fa", "two-factor", "hacked", "stolen", "unauthorized"]):
        return "account_security"
    if any(k in t for k in ["wi-fi", "wifi", "bluetooth", "airpods", "airdrop", "no service", "carrier", "cellular", "disconnect", "hotspot"]):
        return "connectivity"
    if any(k in t for k in ["refund", "subscription", "charge", "charged", "billing", "payment", "card declined", "app store purchase", "apple pay"]):
        return "billing_purchase"
    if any(k in t for k in ["ios", "update", "freeze", "freezing", "crash", "glitch", "boot loop", "app", "sync", "bug", "lag", "restore"]):
        return "software_bug"
    if any(k in t for k in ["compatible", "compatibility", "specs", "applecare", "warranty", "trade in", "trade-in", "work with"]):
        return "product_inquiry"
    if any(k in t for k in ["thank", "thanks", "great", "worst", "terrible", "awful", "service", "store", "complaint", "kudos"]):
        return "general_feedback"
    return "other"

def determine_escalation_heuristically(text: str, intent: str):
    """Determine whether a case should escalate, returning (escalate, trigger, reason)."""
    t = text.lower()
    if any(k in t for k in ["swell", "smoke", "fire", "exploded", "sparks", "melted", "burning hot", "puffy"]):
        return True, "safety_hazard", "Physical safety concern (thermal/swelling/fire risk)"
    if any(k in t for k in ["lawsuit", "attorney", "lawyer", "sue", "legal counsel", "ftc", "better business bureau", "bbb"]):
        return True, "legal_threat", "Legal action or regulatory complaint"
    if any(k in t for k in ["human", "real person", "manager", "supervisor", "representative", "speak to someone"]):
        return True, "human_requested", "Customer explicitly requested human specialist or manager"
    if intent == "account_security":
        return True, "account_security", "Account security and credential integrity requires authenticated support"
    if intent == "billing_purchase" and any(k in t for k in ["unauthorized", "fraud", "stolen", "dispute", "double"]):
        return True, "financial_dispute", "Financial dispute or unauthorized billing requires specialist review"
    return False, "standard_support", "Standard support inquiry suitable for automated guidance"

def compile_golden_eval_dataset():
    logger.info("Compiling verified Golden Evaluation Set from human annotations...")

    if not MANUAL_ANNOTATIONS_PATH.exists():
        logger.info(f"Manual annotations not found at {MANUAL_ANNOTATIONS_PATH}. Initializing candidate sampling...")
        from scripts.sample_candidates import sample_candidates
        sample_candidates()
        raise FileNotFoundError(
            f"Please run 'python scripts/label_golden_set.py' to complete manual annotations at {MANUAL_ANNOTATIONS_PATH}"
        )

    eval_items = []
    with open(MANUAL_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))

    assert len(eval_items) == 200, f"Expected exactly 200 golden examples, got {len(eval_items)}"

    # Validate provenance and human ground truth
    required_provenance = [
        "example_id", "conversation_id", "customer_tweet_id", "customer_text",
        "context", "source", "split", "intent", "expected_escalation",
        "difficulty", "annotator", "annotator_type"
    ]

    intent_counts = {k: 0 for k in INTENT_NAMES}
    for item in eval_items:
        for key in required_provenance:
            if key not in item:
                raise ValueError(f"Golden item {item.get('example_id')} missing required provenance field: '{key}'")

        if item["annotator_type"] != "human_single_annotator":
            raise ValueError(f"Dishonest or invalid annotator identity in item {item.get('example_id')}: {item.get('annotator_type')}")

        # Ensure compatibility fields for evaluator
        item["customer_message"] = item["customer_text"]
        item["ground_truth_intent"] = item["intent"]
        item["ground_truth_escalation"] = "escalate" if item["expected_escalation"] else "auto_handle"
        intent_counts[item["intent"]] += 1

    # Validate intent coverage: every intent must have at least 5 examples
    for intent_name, count in intent_counts.items():
        if count < 5:
            raise ValueError(f"Intent '{intent_name}' has insufficient coverage ({count} < 5 examples)!")

    # Write to final GOLDEN_EVAL_PATH
    with open(GOLDEN_EVAL_PATH, "w", encoding="utf-8") as f:
        for it in eval_items:
            f.write(json.dumps(it) + "\n")

    logger.info(f"Saved {len(eval_items)} verified golden evaluation examples to {GOLDEN_EVAL_PATH}")

    # Automated leakage check against retrieval corpus
    logger.info("Executing automated leakage detection against retrieval corpus...")
    retrieval_corpus = []
    if RETRIEVAL_CORPUS_JSONL_PATH.exists():
        with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    retrieval_corpus.append(json.loads(line))

    leakage_result = check_evaluation_leakage(retrieval_corpus, eval_items)
    print("\n" + format_leakage_report(leakage_result) + "\n")

    if leakage_result["status"] != "PASS":
        raise ValueError(f"CRITICAL: Evaluation leakage detected: {leakage_result}")

    # Summary report
    esc_counts = {}
    source_counts = {}
    for it in eval_items:
        esc_counts[it["ground_truth_escalation"]] = esc_counts.get(it["ground_truth_escalation"], 0) + 1
        source_counts[it["source"]] = source_counts.get(it["source"], 0) + 1

    print("==================================================")
    print("GOLDEN EVALUATION SET COMPILED")
    print("==================================================")
    print(f"Total Examples:      {len(eval_items)}")
    print(f"Human-Labelled:      {len(eval_items)} (annotator: human_single_annotator)")
    print(f"Sources:             {source_counts}")
    print(f"Escalation Split:    {esc_counts}")
    print("Intent Distribution:")
    for k in INTENT_NAMES:
        print(f"  {k:<18}: {intent_counts.get(k, 0)}")
    print("==================================================")

if __name__ == "__main__":
    compile_golden_eval_dataset()
