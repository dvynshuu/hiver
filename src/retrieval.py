import os
import sys
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CORPUS_INDEX_PATH, RAG_TOP_K

logger = logging.getLogger(__name__)

class HistoricalRetrievalEngine:
    """
    Retrieves historically resolved AppleSupport conversations using TF-IDF + cosine similarity.
    Grounds the AI agent's response generation in verified historical resolutions.
    """

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus: List[Dict[str, Any]] = corpus or []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

        if self.corpus:
            self._fit()

    def _fit(self):
        """Build TF-IDF matrix over historical customer inquiries."""
        if not self.corpus:
            return

        texts = [doc.get("customer_text", "") for doc in self.corpus]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most similar historical customer queries and their brand resolutions.
        Returns list of matched items including evidence IDs (pair_id) and similarity scores.
        """
        if not self.corpus or self.vectorizer is None or self.tfidf_matrix is None:
            return []

        query_clean = query.strip()
        if not query_clean:
            return []

        try:
            query_vec = self.vectorizer.transform([query_clean])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0.0:
                    item = dict(self.corpus[idx])
                    item["similarity_score"] = round(score, 4)
                    item["evidence_id"] = item.get("pair_id", f"case_{idx}")
                    results.append(item)
            return results
        except Exception as e:
            logger.warning(f"Error during retrieval: {e}")
            return []

    def format_retrieval_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        Format retrieved historical conversation pairs as context for LLM generation.
        Retains evidence IDs for auditability.
        """
        if not retrieved_items:
            return "No directly matching historical cases found. Follow Apple support standards."

        context_parts = []
        for i, item in enumerate(retrieved_items, 1):
            eid = item.get("evidence_id", item.get("pair_id", f"case_{i}"))
            cust = item.get("customer_text", "").strip()
            reply = item.get("support_reply", "").strip()
            sim = item.get("similarity_score", 0.0)
            context_parts.append(
                f"[Historical Case #{i} | Evidence ID: {eid}] (Similarity: {sim:.2f})\n"
                f"Customer Query: {cust}\n"
                f"AppleSupport Reply: {reply}"
            )

        return "\n\n".join(context_parts)

    def evaluate_retrieval_metrics(
        self,
        eval_items: List[Dict[str, Any]],
        top_k_levels: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Calculates Recall@k on evaluation queries.
        A retrieval match is deemed relevant if:
        1. Cosine similarity score >= 0.15 AND
        2. Content shares domain relevance with query.
        """
        if not self.corpus or self.vectorizer is None:
            return {f"recall_{k}": 0.0 for k in top_k_levels}

        hits = {k: 0 for k in top_k_levels}
        total = 0

        for item in eval_items:
            query = item.get("customer_message", item.get("customer_text", ""))
            if not query.strip():
                continue

            total += 1
            max_k = max(top_k_levels)
            retrieved = self.retrieve(query, top_k=max_k)

            # Check relevance at each k
            # An item is relevant if it achieves meaningful cosine similarity (>0.15)
            # and shares common key tokens
            query_tokens = set(query.lower().split())

            for k in top_k_levels:
                k_retrieved = retrieved[:k]
                found_relevant = False
                for r in k_retrieved:
                    sim = r.get("similarity_score", 0.0)
                    r_tokens = set(r.get("customer_text", "").lower().split())
                    overlap = len(query_tokens.intersection(r_tokens))
                    if sim >= 0.15 and overlap >= 2:
                        found_relevant = True
                        break
                if found_relevant:
                    hits[k] += 1

        metrics = {}
        for k in top_k_levels:
            rate = round(hits[k] / total, 3) if total > 0 else 0.0
            metrics[f"recall_{k}"] = rate

        return metrics

    def save(self, file_path=CORPUS_INDEX_PATH):
        """Save the retrieval corpus and vectorizer to disk."""
        with open(file_path, "wb") as f:
            pickle.dump({
                "corpus": self.corpus,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix
            }, f)

    @classmethod
    def load(cls, file_path=CORPUS_INDEX_PATH) -> "HistoricalRetrievalEngine":
        """Load from disk if available."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Retrieval index not found at {file_path}")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        engine = cls()
        engine.corpus = data["corpus"]
        engine.vectorizer = data["vectorizer"]
        engine.tfidf_matrix = data["tfidf_matrix"]
        return engine
