import os
import re
import html
import json
import random
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_CSV_PATH,
    CLEANED_CONVERSATIONS_PATH,
    RETRIEVAL_CORPUS_JSONL_PATH,
    CORPUS_INDEX_PATH,
    HELD_OUT_EVAL_POOL_PATH,
    GOLDEN_EVAL_PATH,
    TARGET_BRAND,
    MAX_HISTORY_PAIRS,
    SEED
)

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

def normalize_text(text: str) -> str:
    """
    Aggressively normalize text for leakage detection:
    - Lowercase
    - Strip all punctuation, non-alphanumerics, and whitespace
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower())

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
    clean noise, and save structured conversations with conversation IDs.
    """
    if csv_path is None:
        csv_path = find_raw_csv()

    logger.info(f"Processing customer support conversations from {csv_path}...")

    collected_pairs: List[Dict[str, Any]] = []
    seen_customer_texts: Set[str] = set()

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        apple_replies = chunk[
            (chunk["author_id"] == TARGET_BRAND) &
            (chunk["in_response_to_tweet_id"].notna())
        ]

        if apple_replies.empty:
            continue

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

            if len(clean_cust) < 18 or len(clean_apple) < 18:
                continue

            cust_key = clean_cust.lower()
            if cust_key in seen_customer_texts:
                continue
            seen_customer_texts.add(cust_key)

            cust_id = str(int(row["tweet_id_customer"])) if pd.notna(row.get("tweet_id_customer")) else f"c_{len(collected_pairs)+1}"
            apple_id = str(int(row["tweet_id_apple"])) if pd.notna(row.get("tweet_id_apple")) else f"a_{len(collected_pairs)+1}"

            pair = {
                "pair_id": f"apple_{len(collected_pairs) + 1:05d}",
                "conversation_id": f"conv_{cust_id}",
                "customer_tweet_id": cust_id,
                "apple_tweet_id": apple_id,
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

    # Save all cleaned conversations
    with open(CLEANED_CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        for p in collected_pairs:
            f.write(json.dumps(p) + "\n")

    return collected_pairs

def build_conversation_splits(
    cleaned_pairs: Optional[List[Dict[str, Any]]] = None,
    eval_pool_size: int = 200,
    seed: int = SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits the cleaned conversations into:
    1. train_retrieval_corpus (saved to RETRIEVAL_CORPUS_JSONL_PATH and indexed into CORPUS_INDEX_PATH)
    2. held_out_eval_pool (saved to HELD_OUT_EVAL_POOL_PATH)

    Splitting is performed strictly at the conversation level using a deterministic seed.
    The held-out evaluation pool is NEVER indexed into the retrieval engine.
    """
    if cleaned_pairs is None:
        if CLEANED_CONVERSATIONS_PATH.exists():
            cleaned_pairs = []
            with open(CLEANED_CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "conversation_id" not in item:
                            item["conversation_id"] = f"conv_{item.get('customer_tweet_id', item.get('pair_id'))}"
                        cleaned_pairs.append(item)
        else:
            cleaned_pairs = process_and_extract_apple_conversations()

    # Re-save cleaned pairs with conversation_id if missing
    with open(CLEANED_CONVERSATIONS_PATH, "w", encoding="utf-8") as f:
        for p in cleaned_pairs:
            f.write(json.dumps(p) + "\n")

    # Group by conversation_id to guarantee no conversation-level leakage
    conversations_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in cleaned_pairs:
        cid = p.get("conversation_id", f"conv_{p['pair_id']}")
        conversations_map.setdefault(cid, []).append(p)

    conv_ids = sorted(list(conversations_map.keys()))
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    eval_conv_ids = set(conv_ids[:eval_pool_size])
    train_conv_ids = set(conv_ids[eval_pool_size:])

    held_out_pool: List[Dict[str, Any]] = []
    for cid in eval_conv_ids:
        held_out_pool.extend(conversations_map[cid])

    retrieval_corpus: List[Dict[str, Any]] = []
    for cid in train_conv_ids:
        retrieval_corpus.extend(conversations_map[cid])

    logger.info(
        f"Split complete (seed={seed}): {len(retrieval_corpus)} retrieval pairs "
        f"({len(train_conv_ids)} convs) | {len(held_out_pool)} held-out pairs ({len(eval_conv_ids)} convs)."
    )

    # Save retrieval corpus JSONL
    with open(RETRIEVAL_CORPUS_JSONL_PATH, "w", encoding="utf-8") as f:
        for p in retrieval_corpus:
            f.write(json.dumps(p) + "\n")

    # Save held-out eval pool JSONL
    with open(HELD_OUT_EVAL_POOL_PATH, "w", encoding="utf-8") as f:
        for p in held_out_pool:
            f.write(json.dumps(p) + "\n")

    # Fit and save retrieval engine index strictly on retrieval_corpus
    from src.retrieval import HistoricalRetrievalEngine
    logger.info("Fitting and saving historical retrieval index strictly on train/retrieval corpus...")
    engine = HistoricalRetrievalEngine(corpus=retrieval_corpus)
    engine.save(CORPUS_INDEX_PATH)
    logger.info(f"Retrieval index saved to {CORPUS_INDEX_PATH}")

    return retrieval_corpus, held_out_pool

def check_evaluation_leakage(
    retrieval_corpus: List[Dict[str, Any]],
    eval_set: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Automated leakage detection between retrieval corpus and evaluation set.
    Checks:
    1. Exact customer-text overlap
    2. Normalized customer-text overlap (stripped of punctuation/whitespace)
    3. Conversation ID overlap
    4. Customer Tweet ID overlap

    Returns:
        Dict with overlap counts and PASS/FAIL status.
    """
    retrieval_exact_texts: Set[str] = set()
    retrieval_norm_texts: Set[str] = set()
    retrieval_conv_ids: Set[str] = set()
    retrieval_tweet_ids: Set[str] = set()

    for item in retrieval_corpus:
        raw = item.get("customer_text", "").strip()
        if raw:
            retrieval_exact_texts.add(raw)
            retrieval_norm_texts.add(normalize_text(raw))
        cid = item.get("conversation_id")
        if cid:
            retrieval_conv_ids.add(str(cid))
        tid = item.get("customer_tweet_id")
        if tid:
            retrieval_tweet_ids.add(str(tid))

    exact_overlap = 0
    norm_overlap = 0
    conv_overlap = 0
    tweet_overlap = 0

    leaked_examples = []

    for item in eval_set:
        text = item.get("customer_message", item.get("customer_text", "")).strip()
        cid = str(item.get("conversation_id", ""))
        tid = str(item.get("customer_tweet_id", ""))

        is_exact = text in retrieval_exact_texts
        norm = normalize_text(text)
        is_norm = norm in retrieval_norm_texts if len(norm) > 12 else False
        is_conv = bool(cid and cid in retrieval_conv_ids)
        is_tweet = bool(tid and tid in retrieval_tweet_ids)

        if is_exact:
            exact_overlap += 1
        if is_norm:
            norm_overlap += 1
        if is_conv:
            conv_overlap += 1
        if is_tweet:
            tweet_overlap += 1

        if is_exact or is_norm or is_conv or is_tweet:
            leaked_examples.append({
                "example_id": item.get("example_id", item.get("id")),
                "text": text[:60],
                "exact": is_exact,
                "norm": is_norm,
                "conv": is_conv
            })

    status = "PASS" if (exact_overlap == 0 and norm_overlap == 0 and conv_overlap == 0 and tweet_overlap == 0) else "FAIL"

    return {
        "exact_text_overlap": exact_overlap,
        "normalized_overlap": norm_overlap,
        "conversation_overlap": conv_overlap,
        "tweet_id_overlap": tweet_overlap,
        "status": status,
        "leaked_examples": leaked_examples
    }

def format_leakage_report(leakage_result: Dict[str, Any]) -> str:
    """Format leakage check into standardized terminal output."""
    lines = [
        "Leakage Check",
        "--------------",
        f"Exact text overlap:       {leakage_result['exact_text_overlap']}",
        f"Normalized overlap:       {leakage_result['normalized_overlap']}",
        f"Conversation ID overlap:  {leakage_result['conversation_overlap']}",
        f"Status: {leakage_result['status']}"
    ]
    return "\n".join(lines)

def load_or_build_conversations() -> List[Dict[str, Any]]:
    """Ensure cleaned conversations and splits exist, loading retrieval corpus."""
    if (
        CLEANED_CONVERSATIONS_PATH.exists()
        and RETRIEVAL_CORPUS_JSONL_PATH.exists()
        and CORPUS_INDEX_PATH.exists()
        and HELD_OUT_EVAL_POOL_PATH.exists()
    ):
        logger.info(f"Loading retrieval corpus from {RETRIEVAL_CORPUS_JSONL_PATH}...")
        corpus = []
        with open(RETRIEVAL_CORPUS_JSONL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
        return corpus

    logger.info("Initializing dataset cleaning and conversation-level train/eval splitting...")
    retrieval_corpus, _ = build_conversation_splits(eval_pool_size=200, seed=SEED)
    return retrieval_corpus

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = load_or_build_conversations()
    print(f"Data pipeline ready: {len(corpus)} retrieval pairs indexed.")
