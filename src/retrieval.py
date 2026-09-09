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
    Grounds the AI agent's response generation in historical resolutions.
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
                if score > 0.0:  # only include relevant matches
                    item = dict(self.corpus[idx])
                    item["similarity_score"] = round(score, 4)
                    results.append(item)
            return results
        except Exception as e:
            logger.warning(f"Error during retrieval: {e}")
            return []

    def format_retrieval_context(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """
        Format retrieved historical conversation pairs as context for LLM generation.
        """
        if not retrieved_items:
            return "No directly matching historical cases found. Follow Apple support standards."
        
        context_parts = []
        for i, item in enumerate(retrieved_items, 1):
            cust = item.get("customer_text", "").strip()
            reply = item.get("support_reply", "").strip()
            sim = item.get("similarity_score", 0.0)
            context_parts.append(
                f"[Historical Case #{i}] (Relevance Score: {sim:.2f})\n"
                f"Customer Query: {cust}\n"
                f"AppleSupport Reply: {reply}"
            )
        
        return "\n\n".join(context_parts)

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
