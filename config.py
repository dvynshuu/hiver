import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Global deterministic seed
SEED = 42

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_CSV_PATH = DATA_DIR / "twcs.csv"
CLEANED_CONVERSATIONS_PATH = DATA_DIR / "apple_conversations.jsonl"
RETRIEVAL_CORPUS_JSONL_PATH = DATA_DIR / "retrieval_corpus.jsonl"
CORPUS_INDEX_PATH = DATA_DIR / "retrieval_corpus.pkl"
HELD_OUT_EVAL_POOL_PATH = DATA_DIR / "held_out_eval_pool.jsonl"
GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_set.jsonl"
HUMAN_EVAL_RATINGS_PATH = DATA_DIR / "human_eval_ratings.json"
BENCHMARK_RESULTS_PATH = RESULTS_DIR / "benchmark_metrics.json"
RETRIEVAL_BENCHMARK_PATH = DATA_DIR / "retrieval_benchmark.json"

# Brand Target
TARGET_BRAND = "AppleSupport"

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
AGENT_MODEL_NAME = "gemini-3.6-flash"
JUDGE_MODEL_NAME = "gemini-3.6-flash"
FALLBACK_JUDGE_MODEL = "gemini-2.5-flash"

# Evaluation Rubric Dimensions (1-5 scale)
RUBRIC_DIMENSIONS = [
    "groundedness",
    "helpfulness",
    "relevance",
    "brand_alignment",
    "safety"
]

# Intent Taxonomy: 8 distinct intents derived from customer support on Twitter
# Each intent includes name, definition, positive examples, boundary conditions, and confusable intents.
INTENT_TAXONOMY = {
    "device_issue": {
        "name": "Device Hardware Issue",
        "definition": "Physical device failures, battery degradation, screen flicker/cracks, audio/mic malfunction, charging port issues, thermal conditions, camera hardware.",
        "positive_examples": [
            "My iPhone battery dies in 2 hours after the latest update.",
            "The screen on my MacBook Pro is flickering non stop.",
            "My iPhone 11 speaker has crackling sound during calls.",
            "Lightning port is loose and cable keeps disconnecting."
        ],
        "boundary_conditions": "Apply when the primary defect is physical or electrical components. If a battery drain occurs immediately following an iOS update without physical degradation, but customer asks about battery health/hardware, prioritize device_issue. If phone is stuck in boot loop or frozen screen, classify as software_bug.",
        "confusable_intents": ["software_bug", "connectivity"]
    },
    "software_bug": {
        "name": "Software Bug & OS Glitch",
        "definition": "Operating system crashes, app freezes, iOS/macOS update install failures, boot loops, system daemon errors, storage calculation bugs.",
        "positive_examples": [
            "iOS update bricked my phone, stuck in an endless Apple logo loop.",
            "Music app keeps crashing every time I open my playlist.",
            "Cannot download or update apps from the App Store, getting error 403.",
            "Keyboard lag is terrible, takes 3 seconds for letters to appear."
        ],
        "boundary_conditions": "Apply when the defect is rooted in software code, OS behavior, app execution, or firmware updates. If network fails despite OS functioning, use connectivity. If bug involves financial transaction on App Store, use billing_purchase.",
        "confusable_intents": ["device_issue", "connectivity", "billing_purchase"]
    },
    "account_security": {
        "name": "Account Security & Apple ID",
        "definition": "Apple ID locked, two-factor authentication (2FA) verification code issues, account compromise, password resets, iCloud unauthorized logins.",
        "positive_examples": [
            "My Apple ID has been locked for security reasons and verification code is not arriving.",
            "Someone in another country tried to sign into my iCloud account.",
            "I cannot recover my password because my trusted phone number changed.",
            "My phone number was ported in a SIM swap and I'm locked out of iCloud."
        ],
        "boundary_conditions": "Apply to all authentication, credential, authorization, and identity verification challenges. Almost always demands escalation or routing to iforgot.apple.com.",
        "confusable_intents": ["billing_purchase", "software_bug"]
    },
    "connectivity": {
        "name": "Wireless & Network Connectivity",
        "definition": "Wi-Fi disconnections, Bluetooth pairing failures, AirPods connection drops, AirDrop discoverability, cellular 'No Service', hotspot failures.",
        "positive_examples": [
            "My AirPods Pro keep disconnecting from my Mac every 5 minutes.",
            "iPhone says 'No Service' even after resetting network settings.",
            "Bluetooth cannot discover any devices after updating to macOS.",
            "AirDrop won't find my iPad even though Wi-Fi and Bluetooth are on."
        ],
        "boundary_conditions": "Apply when wireless RF protocols (Wi-Fi, Bluetooth, LTE, 5G, AirDrop, NFC) are failing. If lightning cable or wired headphone jack fails physically, classify as device_issue.",
        "confusable_intents": ["device_issue", "software_bug"]
    },
    "billing_purchase": {
        "name": "Billing, Payments & Purchases",
        "definition": "App Store charges, accidental subscriptions, refund requests, payment method declined, invoice/receipt questions, duplicate transactions.",
        "positive_examples": [
            "I was charged $14.99 for a subscription I canceled last week.",
            "Card declined on App Store even though bank says funds are sufficient.",
            "How do I request a refund for an in-app purchase my child made by mistake?",
            "Apple charged me twice for my iCloud 200GB storage plan."
        ],
        "boundary_conditions": "Apply to transactions, currency, payment cards, invoices, App Store purchases, and subscription management. If an account is locked due to billing fraud, cross-references account_security.",
        "confusable_intents": ["account_security", "product_inquiry"]
    },
    "product_inquiry": {
        "name": "Product Specs, Compatibility & AppleCare",
        "definition": "Pre-purchase questions, feature compatibility, warranty terms, AppleCare coverage, trade-in estimates, release dates, retail availability.",
        "positive_examples": [
            "Does the Apple Pencil 2 work with the iPad 10th generation?",
            "How do I check if my iPhone is still under AppleCare+ warranty?",
            "Will trade-in value be credited immediately if I order online?",
            "What is the difference between M3 Pro and M3 Max chips for video editing?"
        ],
        "boundary_conditions": "Apply to informational inquiries about specifications, compatibility, warranties, and purchase terms before or independent of a hardware defect. If inquiring about broken screen repair under warranty, distinguish whether reporting a defect (device_issue) or asking general policy (product_inquiry).",
        "confusable_intents": ["device_issue", "billing_purchase"]
    },
    "general_feedback": {
        "name": "Customer Feedback & Brand Sentiment",
        "definition": "Brand praise, kudos to agents, general complaints, venting about customer service, policy dissatisfaction without a troubleshooting request.",
        "positive_examples": [
            "Apple support is the best! Thank you to the agent who helped me today.",
            "Terrible customer service at the Regent Street store, waited 2 hours for nothing.",
            "Your repair prices are absolute robbery and anti-consumer.",
            "Huge shoutout to Apple Support for replacing my broken keyboard so quickly!"
        ],
        "boundary_conditions": "Apply when customer expresses emotion, gratitude, or frustration without seeking active troubleshooting steps. If customer describes a specific technical glitch while venting, prioritize the technical intent.",
        "confusable_intents": ["other", "device_issue"]
    },
    "other": {
        "name": "Conversational Fillers & Out-of-Scope",
        "definition": "Greetings, spam, unintelligible text, casual pleasantries, off-topic questions, simple conversational sign-offs.",
        "positive_examples": [
            "Thanks for the reply, have a great weekend!",
            "Lol what even is that.",
            "Can you DM me?",
            "Good morning team, hope you're doing well."
        ],
        "boundary_conditions": "Fallback bucket for inputs that contain no actionable technical complaint, security concern, transaction dispute, or product inquiry.",
        "confusable_intents": ["general_feedback"]
    }
}

INTENT_NAMES = list(INTENT_TAXONOMY.keys())

# Escalation Rules and Thresholds
ESCALATION_CRITICAL_INTENTS = {"account_security", "billing_purchase"}

ESCALATION_KEYWORDS = [
    "hacked", "stolen", "fraud", "scam", "unauthorized charge",
    "locked out", "lawsuit", "legal action", "sue", "attorney",
    "supervisor", "manager", "human agent", "talk to human",
    "unacceptable", "furious", "danger", "exploded", "smoke", "fire"
]

# Retrieval settings
RAG_TOP_K = 5
MAX_HISTORY_PAIRS = 5000
