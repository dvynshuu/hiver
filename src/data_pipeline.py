import os
import re
import html
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import (
    RAW_CSV_PATH,
    CLEANED_CONVERSATIONS_PATH,
    CORPUS_INDEX_PATH,
    TARGET_BRAND,
    MAX_HISTORY_PAIRS
)

from src.retrieval import HistoricalRetrievalEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Known Kaggle cache location on Windows
KAGGLE_CACHE_PATH = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "thoughtvector"
    / "customer-support-on-twitter"
    / "versions"
    / "10"
    / "twcs"
    / "twcs.csv"
)

def clean_tweet(text: str) -> str:
    """
    Clean raw tweet text by:
    - Unescaping HTML entities (&gt; &amp; etc.)
    - Removing @handles
    - Removing URLs (t.co links)
    - Normalizing excess whitespace
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def find_raw_csv() -> Path:
    """Locate the raw twcs.csv file from project data, cache, or via kagglehub download."""
    if RAW_CSV_PATH.exists():
        logger.info(f"Found dataset at project path: {RAW_CSV_PATH}")
        return RAW_CSV_PATH
    
    if KAGGLE_CACHE_PATH.exists():
        logger.info(f"Found dataset in Kaggle cache: {KAGGLE_CACHE_PATH}")
        return KAGGLE_CACHE_PATH

    logger.info("Dataset not found locally. Initiating kagglehub download...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        possible_csv = Path(download_path) / "twcs" / "twcs.csv"
        if possible_csv.exists():
            return possible_csv
        for p in Path(download_path).rglob("*.csv"):
            if "twcs" in p.name.lower():
                return p
        raise FileNotFoundError(f"Could not locate twcs.csv in {download_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to locate or download twcs.csv. Please download it from Kaggle "
            f"and place it at {RAW_CSV_PATH}. Error: {e}"
        )

def process_and_extract_apple_conversations(
    csv_path: Optional[Path] = None,
    max_pairs: int = MAX_HISTORY_PAIRS,
    chunk_size: int = 150000
) -> List[Dict[str, Any]]:
    """
    Stream through the dataset, extract customer-AppleSupport reply pairs,
    clean noise, and save structured conversations.
    """
    if csv_path is None:
        csv_path = find_raw_csv()

    logger.info(f"Processing customer support conversations from {csv_path}...")
    
    collected_pairs: List[Dict[str, Any]] = []
    seen_customer_texts = set()

    # Read in chunks to remain memory efficient
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        # Find AppleSupport replies that are in response to a customer tweet
        apple_replies = chunk[
            (chunk["author_id"] == TARGET_BRAND) &
            (chunk["in_response_to_tweet_id"].notna())
        ]
        
        if apple_replies.empty:
            continue

        # Merge with the chunk to find the corresponding customer tweet
        # or build lookup across chunk
        merged = apple_replies.merge(
            chunk[["tweet_id", "text", "author_id", "inbound"]],
            left_on="in_response_to_tweet_id",
            right_on="tweet_id",
            suffixes=("_apple", "_customer")
        )

        for _, row in merged.iterrows():
            raw_cust = str(row["text_customer"])
            raw_apple = str(row["text_apple"])
            
            clean_cust = clean_tweet(raw_cust)
            clean_apple = clean_tweet(raw_apple)

            # Quality filters: avoid empty, image-only, or ultra-short fragments
            if len(clean_cust) < 18 or len(clean_apple) < 18:
                continue
            
            # Avoid exact duplicate inquiries
            cust_key = clean_cust.lower()
            if cust_key in seen_customer_texts:
                continue
            seen_customer_texts.add(cust_key)

            pair = {
                "pair_id": f"apple_{len(collected_pairs) + 1:05d}",
                "customer_tweet_id": str(int(row["tweet_id_customer"])) if pd.notna(row.get("tweet_id_customer")) else "",
                "apple_tweet_id": str(int(row["tweet_id_apple"])) if pd.notna(row.get("tweet_id_apple")) else "",
                "customer_text": clean_cust,
                "customer_raw": raw_cust,
                "support_reply": clean_apple,
                "support_raw": raw_apple,
                "created_at": str(row.get("created_at", ""))
            }
            collected_pairs.append(pair)

            if len(collected_pairs) >= max_pairs:
                break

        logger.info(f"Chunk {chunk_idx + 1}: Collected {len(collected_pairs)} pairs so far...")
        if len(collected_pairs) >= max_pairs:
            break

    logger.info(f"Finished extraction! Total high-quality pairs collected: {len(collected_pairs)}")

    # Save to JSONL
    logger.info(f"Saving conversations to {CLEANED_CONVERSATIONS_PATH}...")
    with open(CLEANED_CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        for p in collected_pairs:
            f.write(json.dumps(p) + "\n")

    # Fit and save the retrieval index
    logger.info("Building and saving historical retrieval index...")
    retrieval_engine = HistoricalRetrievalEngine(corpus=collected_pairs)
    retrieval_engine.save(CORPUS_INDEX_PATH)
    logger.info(f"Retrieval index successfully saved to {CORPUS_INDEX_PATH}!")

    return collected_pairs

def load_or_build_conversations() -> List[Dict[str, Any]]:
    """Load cleaned conversations from disk or build if missing."""
    if CLEANED_CONVERSATIONS_PATH.exists() and CORPUS_INDEX_PATH.exists():
        logger.info(f"Loading cleaned conversations from {CLEANED_CONVERSATIONS_PATH}...")
        pairs = []
        with open(CLEANED_CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    return process_and_extract_apple_conversations()

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pairs = process_and_extract_apple_conversations()
    print(f"Sample pair extracted successfully ({len(pairs)} total pairs indexed)!")
    print(f"Customer: {pairs[0]['customer_text']}")
    print(f"Apple: {pairs[0]['support_reply']}")

