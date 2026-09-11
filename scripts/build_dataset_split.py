"""
Canonical Dataset Split Generator for @AppleSupport.
Splits cleaned customer support conversations into immutable:
1. data/retrieval_corpus.jsonl (for historical retrieval engine)
2. data/held_out_pool.jsonl (for sampling evaluation and golden set)
3. data/split_manifest.json (immutable record of split metadata and conversation IDs)

Evaluation pipelines must NEVER dynamically regenerate this split.
They must only consume the persisted split files and manifest.
"""
import sys
import json
import random
import datetime
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, SEED, TARGET_BRAND

CLEANED_CONVERSATIONS_PATH = DATA_DIR / "apple_conversations.jsonl"
RETRIEVAL_CORPUS_PATH = DATA_DIR / "retrieval_corpus.jsonl"
HELD_OUT_POOL_PATH = DATA_DIR / "held_out_pool.jsonl"
SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"
CORPUS_INDEX_PATH = DATA_DIR / "retrieval_corpus.pkl"

def build_split(seed: int = SEED, held_out_count: int = 350):
    print(f"Loading cleaned conversations from {CLEANED_CONVERSATIONS_PATH}...")
    if not CLEANED_CONVERSATIONS_PATH.exists():
        from src.data_pipeline import process_and_extract_apple_conversations
        process_and_extract_apple_conversations()

    conversations: List[Dict[str, Any]] = []
    with open(CLEANED_CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "conversation_id" not in item:
                    item["conversation_id"] = f"conv_{item.get('customer_tweet_id', item.get('pair_id'))}"
                conversations.append(item)

    # Group by conversation_id to strictly eliminate conversation-level leakage
    conversations_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in conversations:
        cid = p.get("conversation_id", f"conv_{p['pair_id']}")
        conversations_map.setdefault(cid, []).append(p)

    all_conv_ids = sorted(list(conversations_map.keys()))
    rng = random.Random(seed)
    rng.shuffle(all_conv_ids)

    held_out_cids = sorted(all_conv_ids[:held_out_count])
    retrieval_cids = sorted(all_conv_ids[held_out_count:])

    held_out_pairs: List[Dict[str, Any]] = []
    for cid in held_out_cids:
        held_out_pairs.extend(conversations_map[cid])

    retrieval_pairs: List[Dict[str, Any]] = []
    for cid in retrieval_cids:
        retrieval_pairs.extend(conversations_map[cid])

    print(f"Total conversations: {len(all_conv_ids)}")
    print(f"  Retrieval split: {len(retrieval_pairs)} pairs ({len(retrieval_cids)} conversations)")
    print(f"  Held-out split:  {len(held_out_pairs)} pairs ({len(held_out_cids)} conversations)")

    # Save retrieval corpus
    with open(RETRIEVAL_CORPUS_PATH, "w", encoding="utf-8") as f:
        for p in retrieval_pairs:
            f.write(json.dumps(p) + "\n")

    # Save held-out pool (canonical)
    with open(HELD_OUT_POOL_PATH, "w", encoding="utf-8") as f:
        for p in held_out_pairs:
            f.write(json.dumps(p) + "\n")

    # Write split manifest
    manifest = {
        "seed": seed,
        "source": "thoughtvector/customer-support-on-twitter",
        "brand": TARGET_BRAND,
        "total_conversations": len(all_conv_ids),
        "retrieval_conversations_count": len(retrieval_cids),
        "held_out_conversations_count": len(held_out_cids),
        "retrieval_pairs_count": len(retrieval_pairs),
        "held_out_pairs_count": len(held_out_pairs),
        "retrieval_conversations": retrieval_cids,
        "held_out_conversations": held_out_cids,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    with open(SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {SPLIT_MANIFEST_PATH}")

    # Build and save retrieval index strictly on retrieval_corpus
    from src.retrieval import HistoricalRetrievalEngine
    print("Building historical retrieval index strictly on retrieval corpus...")
    engine = HistoricalRetrievalEngine(corpus=retrieval_pairs)
    engine.save(CORPUS_INDEX_PATH)
    print(f"Saved retrieval index to {CORPUS_INDEX_PATH}")
    print("Dataset split complete. Splits are now immutable.")

if __name__ == "__main__":
    build_split()
