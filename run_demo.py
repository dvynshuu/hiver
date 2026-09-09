import sys
import argparse
from pathlib import Path
from tabulate import tabulate

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure stdout encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import TARGET_BRAND
from src.retrieval import HistoricalRetrievalEngine
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator
from src.escalation_engine import EscalationEngine
from src.llm_judge import LLMJudge

SAMPLE_PRESETS = [
    ("Battery Swelling (Safety Hazard)", "My iPhone battery is swelling up and pushing the screen out of the frame! What should I do?"),
    ("Account Lockout (Security Escalation)", "I am locked out of my Apple ID and the recovery code is going to a phone number I don't have anymore."),
    ("Compatibility Inquiry (Auto-handle)", "Does the 2nd generation Apple Pencil work with the 10th gen iPad?"),
    ("Billing Dispute (Financial Escalation)", "Apple charged me $89.99 for an annual subscription that I canceled 3 days ago. I demand a refund!"),
    ("Wi-Fi Connection Issue (Auto-handle)", "My iPhone connects to Wi-Fi but says 'No Internet Connection' while my laptop works fine."),
    ("App Crash / Bug (Auto-handle)", "Ever since updating to the newest iOS, my Notes app crashes instantly when I open any folder.")
]

def format_box(title: str, content: str) -> str:
    border = "-" * 65
    return f"\n+{border}+\n| {title.upper().ljust(63)} |\n+{border}+\n{content}\n"

def process_query(
    query: str,
    retrieval_engine: HistoricalRetrievalEngine,
    classifier: IntentClassifier,
    generator: ReplyGenerator,
    escalation_engine: EscalationEngine,
    judge: LLMJudge
):
    print("\n" + "=" * 70)
    print(f"  CUSTOMER TWEET: \"{query}\"")
    print("=" * 70)

    # 1. Intent Classification
    intent_res = classifier.classify(query)
    intent = intent_res["intent"]
    confidence = intent_res.get("confidence", 0.85)
    print(f"\n[1] INTENT CLASSIFICATION")
    print(f"    - Classified Intent: {intent.upper()}")
    print(f"    - Confidence Score : {confidence * 100:.1f}%")
    print(f"    - Method           : {intent_res.get('method')}")
    print(f"    - Reasoning        : {intent_res.get('reasoning')}")

    # 2. Escalation Decision
    esc_res = escalation_engine.decide(query, intent=intent, intent_confidence=confidence)
    decision = esc_res["decision"].upper()
    urgency = esc_res.get("urgency", "low").upper()
    print(f"\n[2] ESCALATION DECISION")
    print(f"    - Action           : {'[!] ESCALATE TO HUMAN' if decision == 'ESCALATE' else '[*] AUTO-HANDLE BY AGENT'}")
    print(f"    - Urgency Level    : {urgency}")
    print(f"    - Policy Trigger   : {esc_res.get('trigger')}")
    print(f"    - Stated Reason    : {esc_res.get('reason')}")

    # 3. Grounded Retrieval (RAG)
    retrieved = retrieval_engine.retrieve(query, top_k=2)
    print(f"\n[3] HISTORICAL RESOLUTION GROUNDING (Top Matches from {len(retrieval_engine.corpus)} Cases)")
    if retrieved:
        for idx, match in enumerate(retrieved, 1):
            sim = match.get("similarity_score", 0.0)
            hist_q = match.get("customer_text", "")[:70]
            hist_r = match.get("support_reply", "")[:90]
            print(f"    Match #{idx} (Similarity: {sim:.2f}):")
            print(f"      Q: \"{hist_q}...\"")
            print(f"      A: \"{hist_r}...\"")
    else:
        print("    No direct historical match; applying Apple Support guidelines.")

    # 4. Reply Generation
    gen_res = generator.generate_reply(
        query,
        intent=intent,
        escalation_decision=esc_res["decision"],
        escalation_reason=esc_res["reason"]
    )
    reply = gen_res["reply"]
    print(f"\n[4] DRAFTED AGENT REPLY (Apple Brand Voice)")
    print(f"    \"{reply}\"")
    print(f"    (Length: {len(reply)} chars | Method: {gen_res.get('method')})")


    # 5. LLM Judge Quality Evaluation
    judge_res = judge.evaluate_reply(
        customer_text=query,
        agent_reply=reply,
        intent=intent,
        escalation_decision=esc_res["decision"],
        escalation_reason=esc_res["reason"]
    )
    scores = judge_res["dimension_scores"]
    print(f"\n[5] QUALITY ASSURANCE (LLM-as-a-Judge Rubric)")
    print(f"    - Overall Score : {judge_res['overall_score']} / 5.0")
    dim_table = [[k.capitalize(), f"{v} / 5"] for k, v in scores.items()]
    print(tabulate(dim_table, headers=["Dimension", "Score"], tablefmt="simple_grid"))
    print(f"    - Critique      : {judge_res.get('rationale', judge_res.get('critique'))}")
    print("-" * 70)

def main():
    parser = argparse.ArgumentParser(description="Interactive Demo for AppleSupport AI Agent")
    parser.add_argument("--query", type=str, help="Customer tweet message to evaluate")
    parser.add_argument("--offline", action="store_true", help="Run demo using offline deterministic baseline components")
    args = parser.parse_args()

    # Load resources
    print("Loading models and historical retrieval index...")
    retrieval_engine = HistoricalRetrievalEngine.load()
    classifier = IntentClassifier()
    generator = ReplyGenerator(retrieval_engine=retrieval_engine)
    escalation_engine = EscalationEngine()
    judge = LLMJudge()

    def _eval_and_print(q):
        if args.offline:
            print("\n[Running in Offline Mode: Learned Baseline Classifier + Template Generator]")
            print("=" * 70)
            print(f"  CUSTOMER TWEET: \"{q}\"")
            print("=" * 70)
            # 1. Intent
            intent_res = classifier.classify(q, method="learned")
            intent = intent_res["intent"]
            print(f"\n[1] INTENT CLASSIFICATION (Learned Baseline)")
            print(f"    - Classified Intent: {intent.upper()}")
            print(f"    - Confidence Score : {intent_res['confidence'] * 100:.1f}%")
            print(f"    - Method           : {intent_res['method']}")
            # 2. Escalation
            esc_res = escalation_engine.decide(q, intent=intent, intent_confidence=intent_res["confidence"])
            print(f"\n[2] ESCALATION DECISION")
            print(f"    - Action           : {'[!] ESCALATE TO HUMAN' if esc_res['decision'] == 'escalate' else '[*] AUTO-HANDLE BY AGENT'}")
            print(f"    - Urgency Level    : {esc_res['urgency'].upper()}")
            print(f"    - Policy Trigger   : {esc_res['trigger']}")
            print(f"    - Stated Reason    : {esc_res['reason']}")
            # 3. Retrieval
            retrieved = retrieval_engine.retrieve(q, top_k=2)
            print(f"\n[3] HISTORICAL RESOLUTION GROUNDING ({len(retrieval_engine.corpus)} Cases)")
            if retrieved:
                for idx, m in enumerate(retrieved, 1):
                    print(f"    Match #{idx} (Similarity: {m['similarity_score']:.2f}, ID: {m.get('evidence_id')}):")
                    print(f"      Q: \"{m['customer_text'][:65]}...\"")
                    print(f"      A: \"{m['support_reply'][:85]}...\"")
            # 4. Reply
            reply_res = generator.generate_reply(q, intent=intent, escalation_decision=esc_res["decision"], escalation_reason=esc_res["reason"], method="template")
            print(f"\n[4] DRAFTED AGENT REPLY")
            print(f"    \"{reply_res['reply']}\"")
            print(f"    (Length: {len(reply_res['reply'])} chars | Method: {reply_res['method']})")
            # 5. Judge
            judge_res = judge.evaluate_reply(customer_text=q, agent_reply=reply_res["reply"], intent=intent, escalation_decision=esc_res["decision"], escalation_reason=esc_res["reason"], offline=True)
            print(f"\n[5] QUALITY ASSURANCE (Calibrated Heuristic Judge)")
            print(f"    - Overall Score : {judge_res['overall_score']} / 5.0")
            dim_table = [[k.capitalize(), f"{v} / 5"] for k, v in judge_res["dimension_scores"].items()]
            print(tabulate(dim_table, headers=["Dimension", "Score"], tablefmt="simple_grid"))
            print("-" * 70)
        else:
            process_query(q, retrieval_engine, classifier, generator, escalation_engine, judge)

    if args.query:
        _eval_and_print(args.query)
        return

    print("\n" + "=" * 70)
    print("       WELCOME TO THE @AppleSupport AI AGENT INTERACTIVE DEMO")
    print("=" * 70)
    print("Select a sample scenario or enter your own custom customer tweet.\n")

    while True:
        print("\nPreset Scenarios:")
        for idx, (title, sample) in enumerate(SAMPLE_PRESETS, 1):
            print(f"  [{idx}] {title}")
        print("  [C] Enter custom customer message")
        print("  [Q] Quit")

        choice = input("\nEnter choice [1-6, C, Q]: ").strip()
        if not choice:
            continue
        if choice.lower() == "q":
            print("\nExiting demo. Goodbye!")
            break
        elif choice.lower() == "c":
            cust_msg = input("\nEnter customer tweet: ").strip()
            if cust_msg:
                process_query(cust_msg, retrieval_engine, classifier, generator, escalation_engine, judge)
        elif choice.isdigit() and 1 <= int(choice) <= len(SAMPLE_PRESETS):
            _, cust_msg = SAMPLE_PRESETS[int(choice) - 1]
            process_query(cust_msg, retrieval_engine, classifier, generator, escalation_engine, judge)
        else:
            print("Invalid selection. Please enter 1-6, C, or Q.")

if __name__ == "__main__":
    main()
