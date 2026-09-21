"""
Configuration constants and settings for FAQSense retrieval chatbot.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
FAQ_DATA_PATH = DATA_DIR / "faqs.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"

# Project metadata
APP_NAME = "FAQSense"
APP_SUBTITLE = "Retrieval-Based FAQ Chatbot"
ASSIGNMENT_TITLE = "NLP Mid-2 Assignment"
COMPANY_NAME = "FAQ Knowledge Base"
BADGE_TEXT = "TF-IDF + Cosine Similarity"

# NLP & Retrieval hyperparameters
DEFAULT_THRESHOLD = 0.18
MIN_THRESHOLD = 0.00
MAX_THRESHOLD = 1.00
THRESHOLD_STEP = 0.01
NGRAM_RANGE = (1, 2)  # Unigrams and Bigrams
SUBLINEAR_TF = True   # Apply sublinear tf scaling: 1 + log(tf)
TOP_K_MATCHES = 3     # Educational display of top-k candidates

# Standard responses
FALLBACK_RESPONSE = (
    "Sorry, I couldn't find a sufficiently relevant answer in my FAQ knowledge base. "
    "Try asking your question differently or choose one of the suggested FAQs below."
)

EMPTY_QUERY_RESPONSE = (
    "Please enter a question so I can search the knowledge base for you."
)

# Suggested questions for quick start (University Knowledge Base)
SUGGESTED_QUESTIONS = [
    "When was Dr. Harisingh Gour Vishwavidyalaya established?",
    "How can I apply for admission to the university?",
    "Does the university provide hostel facilities?",
    "Does the university have a Central Library?",
    "Where is Dr. Harisingh Gour Vishwavidyalaya located?",
    "What is the Registrar office email address?",
    "Does the university support PhD research?",
    "Does the university have Wi-Fi?",
]

# UI Palette
THEME = {
    "primary": "#4ECDC4",      # Teal accent
    "primary_dark": "#38B2AC",
    "dark": "#0F172A",         # Deep slate
    "secondary_dark": "#111827",
    "card_bg": "#1E293B",      # Slate card
    "bg_light": "#F8FAFC",     # Crisp page bg
    "text_dark": "#0F172A",
    "text_light": "#F8FAFC",
    "text_muted": "#64748B",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "error": "#EF4444",
    "border": "#E2E8F0",
}
