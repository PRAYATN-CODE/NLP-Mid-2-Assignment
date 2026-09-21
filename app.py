"""
FAQSense - Retrieval-Based FAQ Chatbot
Built with Streamlit, Scikit-Learn (TF-IDF), Cosine Similarity, and NLTK.
Mid-2 NLP Assignment Project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from chatbot.config import (
    APP_NAME,
    APP_SUBTITLE,
    ASSIGNMENT_TITLE,
    BADGE_TEXT,
    DEFAULT_THRESHOLD,
    MIN_THRESHOLD,
    MAX_THRESHOLD,
    THRESHOLD_STEP,
    SUGGESTED_QUESTIONS,
    FAQ_DATA_PATH,
    LOGO_PATH,
    THEME,
)
from chatbot.retrieval import FAQRetrievalBot
from chatbot.preprocessing import get_preprocessing_steps
from utils.helpers import (
    inject_custom_css,
    init_session_state,
    record_query_stats,
    clear_chat_history,
    format_score_badge,
    build_category_distribution_chart,
    build_candidate_scores_chart,
    build_session_trend_chart,
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — {ASSIGNMENT_TITLE}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()
init_session_state()


# -----------------------------------------------------------------------------
# 2. MODEL INITIALIZATION & AUTOMATIC RETRAINING
# -----------------------------------------------------------------------------
def get_data_mtime() -> float:
    """Returns dataset modification timestamp to trigger automated retraining."""
    try:
        return FAQ_DATA_PATH.stat().st_mtime
    except Exception:
        return 0.0


@st.cache_resource(show_spinner="Training TF-IDF Vectorizer on FAQ dataset...")
def load_retrieval_bot(data_mtime: float) -> FAQRetrievalBot:
    """Instantiate and train the FAQRetrievalBot on the FAQ dataset."""
    return FAQRetrievalBot()


try:
    bot = load_retrieval_bot(get_data_mtime())
except Exception as e:
    st.error(f"Error loading FAQ knowledge base: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
with st.sidebar:
    # App brand header (Mid 2 NLP Assignment)
    st.markdown(f"""
    <div class="sidebar-brand">
        <h2>🤖 {APP_NAME}</h2>
        <span>{ASSIGNMENT_TITLE}</span>
        <p style="font-size:0.82rem; color:#94A3B8; margin-top:4px;">{APP_SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)

    # Retrain / Reload Action
    if st.button("🔄 Re-train Model / Reload Data", use_container_width=True):
        st.cache_resource.clear()
        st.success(f"Model successfully re-trained on {bot.total_faqs} FAQs!")
        st.rerun()

    # Knowledge Base Stats
    st.markdown("### 📚 Knowledge Base")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.markdown(f"""
        <div class="stat-card-sidebar">
            <div class="stat-num">{bot.total_faqs}</div>
            <div class="stat-lbl">Curated FAQs</div>
        </div>
        """, unsafe_allow_html=True)
    with col_sb2:
        st.markdown(f"""
        <div class="stat-card-sidebar">
            <div class="stat-num">{len(bot.categories)}</div>
            <div class="stat-lbl">Categories</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Retrieval Settings
    st.markdown("### ⚙️ Retrieval Hyperparameters")
    threshold = st.slider(
        "Similarity Threshold (τ)",
        min_value=MIN_THRESHOLD,
        max_value=MAX_THRESHOLD,
        value=DEFAULT_THRESHOLD,
        step=THRESHOLD_STEP,
        help="Minimum Cosine Similarity required to accept an FAQ match. Higher values demand stricter lexical overlap.",
    )

    st.caption(
        f"**Active Threshold: τ = {threshold:.2f}**\n\n"
        "• **Score ≥ {0:.2f}**: High confidence match ✅\n"
        "• **Score < {0:.2f}**: Closest FAQ retrieved 💡".format(threshold)
    )

    st.markdown("---")

    # Architecture info
    st.markdown("### 🔬 NLP Pipeline")
    st.markdown(
        "<div style='font-size:0.82rem; color:#CBD5E1; line-height:1.6;'>"
        "• <strong>Representation:</strong> TF-IDF (1, 2-grams)<br>"
        "• <strong>Scoring:</strong> Cosine Similarity<br>"
        "• <strong>Scaling:</strong> Sublinear TF (1 + log tf)<br>"
        "• <strong>Deterministic:</strong> 100% Zero Hallucination"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Quick Actions
    st.markdown("### 🛠️ Actions")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        clear_chat_history()
        st.rerun()

    st.markdown(
        "<div style='font-size:0.75rem; color:#64748B; text-align:center; margin-top:14px;'>"
        "NLP Mid-2 Assignment Project<br>"
        "<strong>College Practical Evaluation</strong>"
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# 4. MAIN APP TABS
# -----------------------------------------------------------------------------
tab_chat, tab_explorer, tab_analytics, tab_viva = st.tabs([
    "💬 Chat Assistant",
    "📚 FAQ Knowledge Base",
    "📊 Real-Time Analytics",
    "ℹ️ How It Works & Viva Guide",
])


# =============================================================================
# TAB 1: CHATBOT INTERFACE
# =============================================================================
with tab_chat:
    # Modern Chatbot Header Banner
    st.markdown(f"""
    <div class="chatbot-header">
        <div class="chatbot-header-left">
            <div class="bot-avatar-badge">🤖</div>
            <div>
                <h1 class="chatbot-title">{APP_NAME} Assistant</h1>
                <div class="chatbot-subtitle">Retrieval-Based FAQ Chatbot • {ASSIGNMENT_TITLE}</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <div class="status-pill">
                <div class="status-dot"></div>
                <span>NLP IR Pipeline Online ({bot.total_faqs} FAQs)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Prompt Chips (University queries for 1-click testing)
    st.markdown("<div class='chips-label'>💡 Quick Test Questions:</div>", unsafe_allow_html=True)
    chip_cols = st.columns(4)
    chips = [
        ("🏛️ Establishment", "When was Dr. Harisingh Gour Vishwavidyalaya established?"),
        ("🎓 Admissions", "How can I apply for admission to the university?"),
        ("🏠 Hostels", "Does the university provide hostel facilities?"),
        ("📚 Central Library", "Does the university have a Central Library?"),
        ("🔬 PhD Research", "Does the university support PhD research?"),
        ("📶 Wi-Fi Campus", "Does the university have Wi-Fi?"),
        ("📞 Registrar Email", "What is the Registrar office email address?"),
        ("🏛️ Museum", "Where is Gour Sangrahalaya located?"),
    ]
    for i, (label, query) in enumerate(chips):
        with chip_cols[i % 4]:
            if st.button(label, key=f"chip_btn_{i}", use_container_width=True):
                st.session_state.pending_prompt = query
                st.rerun()

    # Welcome Card if conversation is empty
    if len(st.session_state.messages) == 0:
        st.markdown(f"""
        <div class="welcome-container">
            <div class="welcome-heading">Hello! How can I help you today? 👋</div>
            <div class="welcome-subtext">
                Ask any question in natural language. I will search through <strong>{bot.total_faqs} pre-indexed university FAQs</strong>
                across {len(bot.categories)} domains using <strong>TF-IDF Vector Space Modeling</strong> and <strong>Cosine Similarity</strong>.
            </div>
            <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
                <span class="badge-category">🏛️ Overview</span>
                <span class="badge-category">🎓 Admission</span>
                <span class="badge-category">📚 Academics</span>
                <span class="badge-category">🏠 Hostel</span>
                <span class="badge-category">🔬 Research</span>
                <span class="badge-category">🏢 Facilities</span>
                <span class="badge-category">🏛️ Museum</span>
                <span class="badge-category">📞 Contact</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Render Chat Conversation History
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"

        with st.chat_message(role, avatar=avatar):
            if role == "user":
                st.markdown(f"**{msg['content']}**")
            else:
                # Assistant Answer Message with guaranteed contrast & readability
                st.markdown(f"<div class='assistant-answer'>{msg['content']}</div>", unsafe_allow_html=True)

                # Metadata card
                score = msg.get("score", 0.0)
                found = msg.get("found", False)
                matched_q = msg.get("matched_question")
                category = msg.get("category")
                top_matches = msg.get("top_matches", [])

                if found:
                    badge_html = format_score_badge(score, threshold)
                    st.markdown(f"""
                    <div class="metadata-bar">
                        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                            <span class="badge-category">📂 {category}</span>
                            <span><strong>Matched FAQ:</strong> {matched_q}</span>
                        </div>
                        <div>{badge_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Similarity score meter
                    st.progress(
                        min(1.0, max(0.0, score)),
                        text=f"Confidence Score: {int(round(score * 100))}% (Threshold: {int(round(threshold * 100))}%)"
                    )

                    # Technical inspection expander
                    with st.expander("🔍 View Technical NLP Retrieval Details", expanded=False):
                        st.markdown(f"- **Preprocessed Query:** `{msg.get('preprocessed_query', '')}`")
                        st.markdown(f"- **Active Threshold (τ):** `{threshold:.2f}`")
                        st.markdown("- **Vector Space Model:** Scikit-Learn TF-IDF (1,2-grams) + Cosine Similarity")

                        if top_matches:
                            st.plotly_chart(
                                build_candidate_scores_chart(top_matches, threshold),
                                use_container_width=True,
                                key=f"chart_{msg.get('id', np.random.randint(1000000))}"
                            )
                            # Top candidates table
                            st.markdown("**Candidate FAQ Comparison:**")
                            top_df = pd.DataFrame([
                                {
                                    "Candidate Question": m["question"],
                                    "Category": m["category"],
                                    "Cosine Similarity": f"{m['score']:.4f}",
                                    "Decision": "✅ Selected" if i == 0 else "❌ Lower score"
                                }
                                for i, m in enumerate(top_matches)
                            ])
                            st.dataframe(top_df, use_container_width=True, hide_index=True)
                else:
                    # Moderate / Fallback Card
                    badge_html = format_score_badge(score, threshold)
                    st.markdown(f"""
                    <div class="metadata-bar fallback">
                        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                            <span>💡 <strong>Closest FAQ Match:</strong> <em>"{matched_q if matched_q else 'None'}"</em></span>
                        </div>
                        <div>{badge_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("🔍 View Retrieval Inspection & Comparison", expanded=False):
                        st.markdown(f"""
                        - **Calculated Similarity Score:** `{score:.4f}`
                        - **Active Threshold (τ):** `{threshold:.2f}`
                        - You can lower the **Similarity Threshold (τ)** in the sidebar to accept broader matches.
                        """)
                        if top_matches:
                            st.plotly_chart(
                                build_candidate_scores_chart(top_matches, threshold),
                                use_container_width=True,
                                key=f"fallback_chart_{msg.get('id', np.random.randint(1000000))}"
                            )

    # -------------------------------------------------------------------------
    # Handle Chat Input & Processing
    # -------------------------------------------------------------------------
    prompt = st.chat_input("Ask a question about the university (e.g., 'How can I apply for admission?')...")

    # Check if a pending prompt was clicked via quick prompt chips
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        # 1. Append User Message
        st.session_state.messages.append({
            "id": len(st.session_state.messages) + 1,
            "role": "user",
            "content": prompt,
        })

        # 2. Execute Retrieval Pipeline
        result = bot.retrieve_answer(prompt, threshold=threshold)

        # 3. Record analytics
        record_query_stats(result["found"], result["score"])

        # 4. Append Assistant Response
        st.session_state.messages.append({
            "id": len(st.session_state.messages) + 1,
            "role": "assistant",
            "content": result["answer"],
            "score": result["score"],
            "matched_question": result["matched_question"],
            "category": result["category"],
            "found": result["found"],
            "top_matches": result["top_matches"],
            "preprocessed_query": result["preprocessed_query"],
        })

        # 5. Rerun to cleanly update conversation UI
        st.rerun()


# =============================================================================
# TAB 2: FAQ KNOWLEDGE BASE EXPLORER
# =============================================================================
with tab_explorer:
    st.markdown("### 📚 FAQ Knowledge Base Explorer")
    st.markdown(
        "Browse, filter, and inspect the underlying FAQ dataset loaded from `data/faqs.csv`. "
        "Click **'Ask in Chat'** to test any FAQ immediately."
    )

    faq_df = bot.get_df()

    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_term = st.text_input("🔍 Search FAQs by keyword:", placeholder="e.g., admission, hostel, library, research...")
    with col_filter:
        selected_category = st.selectbox(
            "Filter by Category:",
            options=["All Categories"] + bot.categories
        )

    # Filter DataFrame
    filtered_df = faq_df.copy()
    if selected_category != "All Categories":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]

    if search_term.strip():
        term_clean = search_term.lower().strip()
        filtered_df = filtered_df[
            filtered_df["question"].str.lower().str.contains(term_clean, na=False) |
            filtered_df["answer"].str.lower().str.contains(term_clean, na=False)
        ]

    st.markdown(f"**Showing {len(filtered_df)} of {len(faq_df)} FAQs**")

    # Render filtered FAQs as clean expandable cards
    for idx, row in filtered_df.iterrows():
        with st.expander(f"🔹 **[{row['category']}]** {row['question']}", expanded=False):
            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"**Answer:** {row['answer']}")
            st.markdown(f"*Preprocessed NLP Tokens:* `{row['processed_question']}`")
            if st.button("💬 Test in Chat", key=f"btn_ask_{row['id']}"):
                st.session_state.pending_prompt = row["question"]
                st.rerun()


# =============================================================================
# TAB 3: ANALYTICS DASHBOARD
# =============================================================================
with tab_analytics:
    st.markdown("### 📊 Real-Time Retrieval Analytics")
    st.markdown("Monitor chatbot retrieval performance, success rates, and FAQ domain distribution.")

    # Top Metrics Cards
    c1, c2, c3, c4 = st.columns(4)
    total_q = st.session_state.total_queries
    succ_q = st.session_state.successful_retrievals
    fail_q = st.session_state.failed_retrievals
    avg_score = (
        np.mean(st.session_state.scores_history)
        if st.session_state.scores_history
        else 0.0
    )
    success_rate = (succ_q / total_q * 100) if total_q > 0 else 0.0

    with c1:
        st.metric("Total Questions Asked", total_q)
    with c2:
        st.metric("Successful Retrievals", succ_q, f"{success_rate:.1f}% Rate")
    with c3:
        st.metric("Fallback Count", fail_q)
    with c4:
        st.metric("Avg Similarity Score", f"{avg_score:.2f}")

    st.markdown("---")

    col_ch1, col_ch2 = st.columns(2)
    with col_ch1:
        st.markdown("#### 🍩 FAQ Distribution by Category")
        st.plotly_chart(build_category_distribution_chart(bot.get_df()), use_container_width=True)

    with col_ch2:
        st.markdown("#### 📈 Session Cosine Similarity Trend")
        if st.session_state.scores_history:
            fig_trend = build_session_trend_chart(st.session_state.scores_history, threshold)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No queries submitted in this session yet. Ask questions in the Chat tab to view real-time score trends!")


# =============================================================================
# TAB 4: HOW IT WORKS & VIVA PREP
# =============================================================================
with tab_viva:
    st.markdown("### ℹ️ How FAQSense Works — Academic & Viva Guide")
    st.markdown(f"""
    This project demonstrates a traditional **Information Retrieval (IR)** pipeline using
    **TF-IDF Vectorization** and **Cosine Similarity** trained over **{bot.total_faqs} curated FAQs**.
    Unlike generative LLMs, this architecture is:
    - **100% Deterministic & Verifiable**: Responses are retrieved verbatim from curated institutional data.
    - **Zero Hallucination**: If confidence is low, safe guidance is issued.
    - **Lightweight & Fast**: Runs entirely locally in milliseconds without external APIs or GPU requirements.
    """)

    st.markdown("---")

    # Flowchart visualization
    st.markdown("#### 🔄 The Retrieval Architecture")
    st.markdown(f"""
    ```text
    ┌─────────────────────────┐       ┌──────────────────────────┐
    │ 1. User Natural Query   │       │ Curated FAQ Dataset (CSV)│
    │ "How to get admission?" │       │ {bot.total_faqs} Questions & Answers   │
    └────────────┬────────────┘       └─────────────┬────────────┘
                 │                                  │
                 ▼                                  ▼
    ┌─────────────────────────┐       ┌──────────────────────────┐
    │ 2. Text Preprocessing   │       │ Text Preprocessing       │
    │ Lower, Clean, Stem      │       │ Identical Pipeline       │
    └────────────┬────────────┘       └─────────────┬────────────┘
                 │                                  │
                 ▼                                  ▼
    ┌─────────────────────────┐       ┌──────────────────────────┐
    │ 3. Transform with       │       │ Train TF-IDF Vectorizer  │
    │    Trained Vectorizer   │       │ N-gram range: (1, 2)     │
    └────────────┬────────────┘       └─────────────┬────────────┘
                 │                                  │
                 ▼                                  ▼
            User Vector                       FAQ Vectors Matrix
                 │                                  │
                 └───────────────┬──────────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 4. Cosine Similarity  │
                     │    cos(θ) = (A·B)/(|A||B|)
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ 5. Max Score >= τ ?   │
                     └─────┬───────────┬─────┘
                           │           │
                     YES   │           │   NO
                           ▼           ▼
             ┌────────────────┐     ┌────────────────┐
             │ Retrieve FAQ   │     │ Closest Match  │
             │ Predefined     │     │ or Fallback    │
             │ Answer         │     │ Response       │
             └────────────────┘     └────────────────┘
    ```
    """)

    st.markdown("---")

    # Mathematical Foundations
    st.markdown("#### 📐 Mathematical Foundations for Viva")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 1. Term Frequency (TF)")
        st.latex(r"TF(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0")
        st.caption("Sublinear TF scaling reduces the disproportionate impact of words that repeat many times.")

        st.markdown("##### 2. Inverse Document Frequency (IDF)")
        st.latex(r"IDF(t, D) = \log\left(\frac{1 + |D|}{1 + DF(t)}\right) + 1")
        st.caption("Gives higher weight to rare, discriminative words and lower weight to common words.")

    with col_m2:
        st.markdown("##### 3. TF-IDF Score & Normalization")
        st.latex(r"\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)")
        st.caption("Vectors are L2-normalized so that length of FAQ question does not distort similarity.")

        st.markdown("##### 4. Cosine Similarity Formula")
        st.latex(r"\text{Cosine Similarity}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \cos(\theta)")
        st.caption("Measures angle between query vector q and document vector d; ranges from 0 (orthogonal/unrelated) to 1 (identical).")

    st.markdown("---")

    # Interactive TF-IDF Diagnostic Tool
    st.markdown("#### 🧪 Interactive TF-IDF Diagnostic Playground")
    st.markdown("Type any sample query below to inspect its preprocessed tokens, non-zero TF-IDF vocabulary weights, and live similarity scores in real-time.")

    sample_query = st.text_input("Test query for diagnostic inspection:", value="When was Dr. Harisingh Gour Vishwavidyalaya established?")
    if sample_query.strip():
        diag_res = get_preprocessing_steps(sample_query)
        term_weights = bot.get_query_terms_weights(sample_query)
        eval_res = bot.retrieve_answer(sample_query, threshold=threshold)

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**1. Preprocessing Steps:**")
            st.json({
                "Original": diag_res["raw_text"],
                "Cleaned": diag_res["punct_removed"],
                "Tokens": diag_res["tokens"],
                "Final Preprocessed": diag_res["final_preprocessed"]
            })

            st.markdown("**2. Query TF-IDF Vector (Non-zero terms):**")
            if term_weights:
                weights_df = pd.DataFrame(term_weights)
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No overlapping vocabulary found with the FAQ training corpus.")

        with dcol2:
            st.markdown("**3. Top 3 Candidate FAQ Similarities:**")
            st.plotly_chart(
                build_candidate_scores_chart(eval_res["top_matches"], threshold),
                use_container_width=True,
                key="viva_diag_chart"
            )
            st.markdown(f"**Outcome:** {'✅ Match Accepted' if eval_res['found'] else '💡 Closest Match'}")
            if eval_res["matched_question"]:
                st.success(f"**Best Match:** {eval_res['matched_question']} ({eval_res['score']:.4f})")
            else:
                st.warning(f"Top score ({eval_res['score']:.4f}) is below threshold ({threshold:.2f}).")
