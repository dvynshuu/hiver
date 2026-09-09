import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

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
CORPUS_INDEX_PATH = DATA_DIR / "retrieval_corpus.pkl"
GOLDEN_EVAL_PATH = DATA_DIR / "golden_eval_set.jsonl"
BENCHMARK_RESULTS_PATH = RESULTS_DIR / "benchmark_metrics.json"

# Brand Target
TARGET_BRAND = "AppleSupport"

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
AGENT_MODEL_NAME = "gemini-3.6-flash"
JUDGE_MODEL_NAME = "gemini-3.6-flash"
FALLBACK_JUDGE_MODEL = "gemini-2.5-flash"



# Intent Taxonomy (8 distinct intents derived from customer support on Twitter)
INTENT_TAXONOMY = {
    "device_issue": {
        "description": "Hardware problems, battery drain, physical damage, screen, audio/speaker, charging, camera issues.",
        "examples": [
            "My iPhone battery dies in 2 hours after the latest update.",
            "The screen on my MacBook Pro is flickering non stop.",
            "My iPhone 11 speaker has crackling sound during calls."
        ]
    },
    "software_bug": {
        "description": "OS crashes, app freezes, update installation failures, boot loops, system glitches.",
        "examples": [
            "iOS 17 update bricked my phone, it is stuck in an endless boot loop.",
            "Music app keeps crashing every time I open my playlist.",
            "Cannot download or update apps from the App Store, getting error 403."
        ]
    },
    "account_security": {
        "description": "Apple ID locked, 2FA issues, compromised accounts, password resets, iCloud sign-in errors.",
        "examples": [
            "My Apple ID has been locked for security reasons and verification code is not arriving.",
            "Someone in another country tried to sign into my iCloud.",
            "I cannot recover my password because my trusted phone number changed."
        ]
    },
    "connectivity": {
        "description": "Wi-Fi dropouts, Bluetooth pairing failures, AirPods connection, AirDrop, cellular data/no service.",
        "examples": [
            "My AirPods Pro keep disconnecting from my Mac every 5 minutes.",
            "iPhone says 'No Service' even after resetting network settings.",
            "Bluetooth cannot discover any devices after updating to macOS Sonoma."
        ]
    },
    "billing_purchase": {
        "description": "App Store charges, accidental subscriptions, refund requests, payment method declined, invoice questions.",
        "examples": [
            "I was charged $14.99 for a subscription I canceled last week.",
            "Card declined on App Store even though bank says funds are sufficient.",
            "How do I request a refund for an in-app purchase my child made by mistake?"
        ]
    },
    "product_inquiry": {
        "description": "Pre-purchase questions, feature compatibility, warranty terms, AppleCare coverage, release dates.",
        "examples": [
            "Does the Apple Pencil 2 work with the iPad 10th generation?",
            "How do I check if my iPhone is still under AppleCare+ warranty?",
            "Will trade-in value be credited immediately if I order online?"
        ]
    },
    "general_feedback": {
        "description": "Brand praise, general complaints, venting about customer service, policy dissatisfaction.",
        "examples": [
            "Apple support is the best! Thank you to the agent who helped me today.",
            "Terrible customer service at the Regent Street store, waited 2 hours for nothing.",
            "Your repair prices are absolute robbery."
        ]
    },
    "other": {
        "description": "Messages that do not fit the above categories, spam, unintelligible messages, simple conversational fillers.",
        "examples": [
            "Thanks for the reply, have a great weekend!",
            "Lol what even is that.",
            "Can you DM me?"
        ]
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
