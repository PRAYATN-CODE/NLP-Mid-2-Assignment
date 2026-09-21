"""
FAQSense Chatbot Package
A retrieval-based NLP FAQ chatbot using TF-IDF and Cosine Similarity.
"""

from chatbot.retrieval import FAQRetrievalBot
from chatbot.preprocessing import preprocess_text

__all__ = ["FAQRetrievalBot", "preprocess_text"]
