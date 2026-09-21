# FAQSense — Retrieval-Based FAQ Chatbot

> **College Mid-2 NLP Assignment Project**  
> A complete, production-ready, retrieval-based chatbot built using traditional Natural Language Processing (NLP) techniques: **TF-IDF Vectorization** and **Cosine Similarity**.  
> **100% Deterministic • Zero Generative LLM Hallucinations • Sub-millisecond Local Inference**

---

## 📌 Project Overview

**FAQSense** is an educational and practical implementation of an Information Retrieval (IR) chatbot for an academic NLP assignment. It answers user questions by searching through a pre-indexed FAQ dataset of **55+ questions and answers** across 10 service categories.

Unlike generative AI chatbots (such as ChatGPT, Gemini, or Claude) which synthesize answers and can suffer from hallucinations, FAQSense uses **vector space modeling** to strictly retrieve curated, human-verified answers.

---

## 🚀 Key Features

* **Traditional NLP Pipeline**: Fully transparent, deterministic retrieval using Scikit-Learn and NLTK.
* **Symmetrical Text Preprocessing**: Consistent lowercase transformation, punctuation removal, whitespace normalization, and stemming.
* **TF-IDF Vector Space Model**: Uses unigrams and bigrams (`ngram_range=(1, 2)`) with sublinear term-frequency scaling.
* **Cosine Similarity Scoring**: Computes directional vector alignment to find the most relevant FAQ.
* **Configurable Similarity Threshold ($\tau$)**: Dynamic interactive slider ($0.00$ to $1.00$, default $0.20$) in the sidebar to control matching sensitivity and prevent spurious answers.
* **Graceful Fallback Mechanism**: Detects out-of-domain and low-confidence queries safely.
* **Interactive Streamlit Dashboard**:
  * **💬 Chat**: Natural conversation stream with similarity badges, category tags, candidate score charts, and retrieval inspection expanders.
  * **📚 FAQ Explorer**: Searchable and filterable knowledge base with one-click "Ask in Chat".
  * **📊 Real-Time Analytics**: Session metrics, success rates, Plotly category donut charts, and score trends.
  * **ℹ️ How It Works & Viva Guide**: Interactive TF-IDF diagnostic playground, mathematical formulas, and viva questions.
* **Comprehensive Test Suite**: Automated unit tests covering exact matches, paraphrasing, edge cases, and threshold behavior.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[User Natural Query<br/>'When are you open?'] --> B[NLP Preprocessing<br/>Lower, Clean, Stem]
    FAQ[(FAQ Dataset CSV<br/>55 Knowledge Base FAQs)] --> C[NLP Preprocessing<br/>Identical Pipeline]
    C --> D[Fit TF-IDF Vectorizer<br/>Unigrams + Bigrams]
    D --> E[Precomputed FAQ<br/>TF-IDF Matrix]
    B --> F[Transform Query to Vector]
    F --> G[Cosine Similarity Computation<br/>cos θ = A · B / |A||B|]
    E --> G
    G --> H[Find Highest Similarity Score<br/>Best Match Index]
    H --> I{Score >= Threshold?}
    I -->|Yes| J[Retrieve Predefined FAQ Answer]
    I -->|No| K[Trigger Safe Fallback Response]
    J --> L[Streamlit Response Card<br/>+ Metric Badges & Chart]
    K --> L
```

---

## 📂 Project Structure

```text
NLP MID-2 ASSIGNMENT/
│
├── app.py                     # Main Streamlit web application
│
├── chatbot/                   # Core NLP & Retrieval Engine
│   ├── __init__.py            # Package export
│   ├── config.py              # Hyperparameters, paths, and UI palette
│   ├── preprocessing.py       # Symmetrical text cleaning & stemming
│   └── retrieval.py           # TF-IDF Vectorizer & Cosine Similarity bot
│
├── data/
│   └── faqs.csv               # 55 Curated FAQs across 10 categories
│
├── assets/
│   └── logo.png               # FAQSense brand emblem
│
├── utils/                     # UI and Dashboard Helpers
│   ├── __init__.py
│   └── helpers.py             # Custom CSS, session management, Plotly charts
│
├── tests/
│   └── test_retrieval.py      # Automated unit test suite
│
├── requirements.txt           # Minimal Python dependencies
├── README.md                  # Project documentation & viva guide
└── .gitignore                 # Standard exclusions
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- Git (optional).

### 2. Install Dependencies
In your terminal, navigate to the project directory:

```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
Verify that all algorithms and edge cases pass:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧠 How the Algorithm Works (Academic & Viva Explanation)

### 1. Text Preprocessing
Both the FAQ questions in the dataset and the incoming user questions undergo an identical pipeline:
1. **Case Normalization**: `text.lower()`
2. **Punctuation & Noise Removal**: Strips URLs, punctuation marks, and non-alphanumeric characters.
3. **Regex Tokenization**: Splits words at boundary boundaries `\b[a-z0-9]+\b`.
4. **Stopword Filtering**: Removes high-frequency English functional words while preserving question interrogatives (*how, when, where, why*).
5. **Morphological Stemming**: Normalizes word forms (e.g., "visiting" $\to$ "visit", "hours" $\to$ "hour").

### 2. TF-IDF Vectorization
**TF-IDF** represents documents as numerical vectors in a high-dimensional vector space:
* **Term Frequency (TF)**:
  $$\text{TF}(t, d) = 1 + \log(f_{t, d})$$
  Measures the frequency of term $t$ in document $d$. Sublinear scaling reduces disproportionate weight from high frequencies.
* **Inverse Document Frequency (IDF)**:
  $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
  Measures how informative a term is across the entire corpus $D$. Rare words receive higher IDF weights.
* **TF-IDF Vector**:
  $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

### 3. Cosine Similarity
Cosine similarity evaluates the cosine of the angle between the user query vector $\vec{q}$ and each FAQ question vector $\vec{d}_i$:

$$\text{Cosine Similarity}(\vec{q}, \vec{d}_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\|_2 \|\vec{d}_i\|_2} = \frac{\sum_{k} q_k d_{ik}}{\sqrt{\sum_{k} q_k^2} \sqrt{\sum_{k} d_{ik}^2}}$$

- **Score $= 1.0$**: Perfect alignment (identical terms).
- **Score $= 0.0$**: Orthogonal vectors (no overlapping vocabulary).

### 4. Similarity Threshold Gating ($\tau$)
- If $\max(\text{similarity}) \ge \tau$: Return the best-matching FAQ answer.
- If $\max(\text{similarity}) < \tau$: Return fallback response.

---

## 🧪 Demonstration Examples for Viva

| Query Type | Sample Input | Expected Behavior | Similarity Score |
| :--- | :--- | :--- | :--- |
| **Exact FAQ** | *"What are your working hours?"* | Returns office hours answer | $\approx 0.85 - 1.00$ |
| **Paraphrased Query** | *"When can I visit your office?"* | Matches FAQ 52 (office visits) | $\approx 0.50 - 0.75$ |
| **Technical Support** | *"How can I reset my password?"* | Matches FAQ 6 (Authentication) | $\approx 0.80 - 0.95$ |
| **Irrelevant / Out of Domain** | *"Who won the cricket match?"* | Safely triggers Fallback Response | $< 0.10$ |
| **Edge Case: Empty Input** | `""` | Prompts user to enter a question | $0.00$ |

---

## 📊 Traditional Retrieval vs Generative AI (LLMs)

| Feature | FAQSense (TF-IDF Retrieval) | Generative LLMs (e.g. GPT-4) |
| :--- | :--- | :--- |
| **Factual Accuracy** | **100% (Verbatim curated answers)** | Risk of hallucination |
| **API Costs & Dependencies** | **$0.00 (Zero external APIs)** | Ongoing token/API charges |
| **Latency** | **< 20 milliseconds** | 500 ms – 3,000 ms |
| **Hardware Requirements** | Runs on any modest laptop/CPU | Requires GPU or cloud connection |
| **Explainability** | Full mathematical transparency | Black-box neural reasoning |

---

## 🎯 Viva Q&A Cheat Sheet for Students

**Q1: Why use TF-IDF instead of simple keyword matching?**  
*Answer:* Simple keyword matching gives equal weight to all words. TF-IDF downweights ubiquitous words (like "the", "is") and boosts discriminative words (like "password", "refund", "invoice").

**Q2: What is the purpose of bigrams (`ngram_range=(1, 2)`)?**  
*Answer:* Bigrams preserve word order for critical phrases such as "reset password", "credit card", and "working hours" which carry distinct semantics beyond their individual single words.

**Q3: Why Cosine Similarity instead of Euclidean Distance?**  
*Answer:* Euclidean distance is sensitive to document length (a longer FAQ would have a larger distance even if discussing the same topic). Cosine similarity normalizes length and focuses exclusively on directional angle.

**Q4: What is the role of the similarity threshold?**  
*Answer:* Without a threshold, `argmax` would always force a response even if the highest similarity is only 0.02. The threshold prevents false positives and ensures high precision.

---

## 🔮 Future Enhancements
* Dense vector representations via Sentence-Transformers / BERT embeddings.
* Hybrid search (sparse BM25 + dense neural bi-encoder).
* SQLite / PostgreSQL database backend for dynamic FAQ authoring.
* Multi-lingual question retrieval.
