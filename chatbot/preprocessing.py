"""
Text Preprocessing Module for FAQSense Chatbot.

Provides a clean, transparent, and symmetrical NLP preprocessing pipeline
applied to both FAQ questions and user input questions.
Uses NLTK PorterStemmer and Stopwords with 100% offline reliability.
"""

import re
import string
from typing import List, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Global cached tools
_STOPWORDS_SET = None
_STEMMER = PorterStemmer()
_LEMMATIZER = None


def _initialize_nlp():
    """Safely initialize NLTK stopwords and lemmatizer/stemmer without network blocking."""
    global _STOPWORDS_SET, _LEMMATIZER
    try:
        _STOPWORDS_SET = set(stopwords.words("english"))
    except Exception:
        # Fallback built-in stop words
        _STOPWORDS_SET = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves"
        }

    # Check if WordNet is available locally without hanging
    try:
        nltk.data.find("corpora/wordnet")
        _LEMMATIZER = WordNetLemmatizer()
    except LookupError:
        _LEMMATIZER = None


_initialize_nlp()


def clean_text(text: str) -> str:
    """
    Perform baseline textual cleaning:
    1. Lowercase conversion
    2. URL removal
    3. Punctuation removal
    4. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\bwi\s*[-_]?\s*fi\b", "wifi", text)
    text = re.sub(r"\bph\.?\s*d\b", "phd", text)
    text = re.sub(r"\bdr\.\s*", "dr ", text)
    text = re.sub(r"\bpin\s+code\b", "pincode", text)
    punct_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text = text.translate(punct_table)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize string into individual word tokens using regex word boundaries.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return re.findall(r"\b[a-z0-9]+\b", cleaned)


def normalize_token(token: str) -> str:
    """
    Normalize token using WordNet lemmatizer if available, or PorterStemmer.
    """
    if _LEMMATIZER:
        try:
            return _LEMMATIZER.lemmatize(token)
        except Exception:
            pass
    # Reliable PorterStemmer fallback
    try:
        return _STEMMER.stem(token)
    except Exception:
        return token


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True
) -> str:
    """
    Complete NLP preprocessing pipeline:
    Input String -> Lowercase -> Remove Punctuation -> Tokenize
                 -> (Optional) Remove Stopwords -> (Optional) Stem/Lemmatize
                 -> Normalized String
    """
    tokens = tokenize(text)
    if not tokens:
        return ""

    processed_tokens: List[str] = []
    # Filter standard stopwords, preserving critical negation
    keep_words = {"not", "no"}

    for token in tokens:
        if remove_stopwords and _STOPWORDS_SET and token in _STOPWORDS_SET:
            if token not in keep_words:
                continue

        if apply_stemming:
            token = normalize_token(token)

        processed_tokens.append(token)

    return " ".join(processed_tokens)


def get_preprocessing_steps(raw_text: str) -> Dict[str, Any]:
    """
    Returns step-by-step intermediate transformations for educational/viva inspection.
    """
    lowered = raw_text.lower() if raw_text else ""
    punct_removed = clean_text(raw_text)
    tokens = tokenize(raw_text)
    normalized = [normalize_token(t) for t in tokens]
    final_text = preprocess_text(raw_text)

    return {
        "raw_text": raw_text,
        "lowered": lowered,
        "punct_removed": punct_removed,
        "tokens": tokens,
        "normalized_tokens": normalized,
        "final_preprocessed": final_text,
    }
