"""
Retrieval Engine Module for FAQSense Chatbot.

Implements traditional Information Retrieval (IR) pipeline:
FAQ Dataset -> Preprocessing -> TF-IDF Vectorization -> Cosine Similarity -> Threshold -> Answer Retrieval.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot.config import (
    FAQ_DATA_PATH,
    DEFAULT_THRESHOLD,
    NGRAM_RANGE,
    SUBLINEAR_TF,
    TOP_K_MATCHES,
    FALLBACK_RESPONSE,
    EMPTY_QUERY_RESPONSE,
)
from chatbot.preprocessing import preprocess_text, get_preprocessing_steps


class FAQRetrievalBot:
    """
    Retrieval-based FAQ Chatbot using Scikit-Learn TF-IDF Vectorizer
    and Cosine Similarity.
    """

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize and fit the retrieval model on the FAQ dataset."""
        self.data_path = Path(data_path) if data_path else FAQ_DATA_PATH
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.faq_matrix = None
        self.is_ready: bool = False

        self._load_and_train()

    def _load_and_train(self) -> None:
        """Load FAQ dataset, preprocess questions, and compute TF-IDF matrix."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"FAQ dataset file not found at: {self.data_path}")

        # Load CSV
        self.df = pd.read_csv(self.data_path)
        required_cols = {"id", "category", "question", "answer"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"FAQ CSV must contain columns: {required_cols}. Found: {self.df.columns.tolist()}")

        # Ensure no nulls in text fields
        self.df["question"] = self.df["question"].fillna("").astype(str)
        self.df["answer"] = self.df["answer"].fillna("").astype(str)
        self.df["category"] = self.df["category"].fillna("General").astype(str)

        # Preprocess all FAQ questions symmetrically
        self.df["processed_question"] = self.df["question"].apply(preprocess_text)

        # Enriched search corpus: question weighted 2x + answer content
        self.df["search_corpus"] = (
            self.df["question"] + " " + self.df["question"] + " " + self.df["answer"]
        ).apply(preprocess_text)

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            sublinear_tf=SUBLINEAR_TF,
            token_pattern=r"\b[a-zA-Z0-9]+\b"
        )

        # Fit vectorizer on search corpus and transform to TF-IDF matrix
        self.faq_matrix = self.vectorizer.fit_transform(self.df["search_corpus"])
        self.is_ready = True

    def retrieve_answer(
        self,
        user_question: str,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = TOP_K_MATCHES
    ) -> Dict[str, Any]:
        """
        Retrieves the best matching answer for a user query.

        Args:
            user_question: Raw natural-language question from user.
            threshold: Minimum cosine similarity score required for match.
            top_k: Number of top candidate FAQs to return for inspection.

        Returns:
            Structured dictionary with answer, similarity score, match details, etc.
        """
        if not self.is_ready or self.vectorizer is None or self.faq_matrix is None:
            return {
                "answer": "Retrieval model is not ready. Please verify dataset loading.",
                "score": 0.0,
                "matched_question": None,
                "category": None,
                "found": False,
                "top_matches": [],
                "preprocessed_query": "",
                "threshold_used": threshold,
            }

        # Check for empty or whitespace query
        trimmed_query = user_question.strip() if user_question else ""
        if not trimmed_query:
            return {
                "answer": EMPTY_QUERY_RESPONSE,
                "score": 0.0,
                "matched_question": None,
                "category": None,
                "found": False,
                "top_matches": [],
                "preprocessed_query": "",
                "threshold_used": threshold,
            }

        # Preprocess user query with same pipeline
        processed_query = preprocess_text(trimmed_query)

        # If preprocessing eliminates all tokens (e.g., pure punctuation), fallback
        if not processed_query:
            return {
                "answer": FALLBACK_RESPONSE,
                "score": 0.0,
                "matched_question": None,
                "category": None,
                "found": False,
                "top_matches": [],
                "preprocessed_query": processed_query,
                "threshold_used": threshold,
            }

        # Transform user query using the trained TF-IDF vectorizer
        query_vector = self.vectorizer.transform([processed_query])

        # Compute Cosine Similarity between user query and all FAQ vectors
        similarity_scores = cosine_similarity(query_vector, self.faq_matrix).flatten()

        # Handle all-zero similarity (no overlapping vocabulary)
        max_score = float(np.max(similarity_scores)) if len(similarity_scores) > 0 else 0.0
        best_idx = int(np.argmax(similarity_scores)) if len(similarity_scores) > 0 else 0

        # Sort candidate indices by similarity score descending
        sorted_indices = np.argsort(similarity_scores)[::-1]
        top_indices = sorted_indices[:min(top_k, len(sorted_indices))]

        top_matches: List[Dict[str, Any]] = []
        for idx in top_indices:
            score_val = float(similarity_scores[idx])
            top_matches.append({
                "id": int(self.df.iloc[idx]["id"]),
                "category": str(self.df.iloc[idx]["category"]),
                "question": str(self.df.iloc[idx]["question"]),
                "answer": str(self.df.iloc[idx]["answer"]),
                "score": round(score_val, 4),
            })

        # Threshold Decision
        if max_score >= threshold:
            best_row = self.df.iloc[best_idx]
            return {
                "answer": str(best_row["answer"]),
                "score": round(max_score, 4),
                "matched_question": str(best_row["question"]),
                "category": str(best_row["category"]),
                "found": True,
                "top_matches": top_matches,
                "preprocessed_query": processed_query,
                "threshold_used": threshold,
            }
        elif max_score >= 0.05 and top_matches:
            # Partial/moderate match - return the closest answer so user question is answered!
            best_row = self.df.iloc[best_idx]
            return {
                "answer": str(best_row["answer"]),
                "score": round(max_score, 4),
                "matched_question": str(best_row["question"]),
                "category": str(best_row["category"]),
                "found": False,
                "top_matches": top_matches,
                "preprocessed_query": processed_query,
                "threshold_used": threshold,
            }
        else:
            # Completely out of scope query (zero or near-zero similarity)
            return {
                "answer": FALLBACK_RESPONSE,
                "score": round(max_score, 4),
                "matched_question": top_matches[0]["question"] if top_matches else None,
                "category": top_matches[0]["category"] if top_matches else None,
                "found": False,
                "top_matches": top_matches,
                "preprocessed_query": processed_query,
                "threshold_used": threshold,
            }

    def get_query_terms_weights(self, query: str) -> List[Dict[str, Any]]:
        """
        Returns the non-zero TF-IDF terms and their computed weights in the query vector.
        Useful for viva demo and educational inspection.
        """
        if not self.is_ready or self.vectorizer is None:
            return []

        processed = preprocess_text(query)
        if not processed:
            return []

        vec = self.vectorizer.transform([processed])
        feature_names = self.vectorizer.get_feature_names_out()
        cx = vec.tocoo()

        weights = []
        for _, col, value in zip(cx.row, cx.col, cx.data):
            weights.append({
                "term": feature_names[col],
                "tfidf_weight": round(float(value), 4)
            })

        weights.sort(key=lambda x: x["tfidf_weight"], reverse=True)
        return weights

    @property
    def total_faqs(self) -> int:
        """Returns total number of FAQs loaded."""
        return len(self.df)

    @property
    def categories(self) -> List[str]:
        """Returns list of unique FAQ categories."""
        if "category" in self.df.columns:
            return sorted(self.df["category"].unique().tolist())
        return []

    def get_df(self) -> pd.DataFrame:
        """Returns the underlying FAQ dataframe."""
        return self.df.copy()
