"""
Helper functions, CSS styling, session management, and visualization builders
for the FAQSense Retrieval Chatbot (NLP Mid-2 Assignment).
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from chatbot.config import THEME, APP_NAME


def inject_custom_css() -> None:
    """Injects modern, polished CSS styles into the Streamlit application."""
    custom_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-heading: 'Outfit', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --primary: {THEME['primary']};
        --primary-dark: {THEME['primary_dark']};
        --bg-dark: {THEME['dark']};
        --card-bg: {THEME['card_bg']};
        --text-dark: {THEME['text_dark']};
        --text-muted: {THEME['text_muted']};
        --success: {THEME['success']};
        --warning: {THEME['warning']};
        --error: {THEME['error']};
    }}

    html, body, [class*="css"] {{
        font-family: var(--font-main);
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--font-heading);
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    code, pre {{
        font-family: var(--font-mono) !important;
    }}

    /* Main chat top banner */
    .chatbot-header {{
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 16px;
        padding: 20px 24px;
        color: #F8FAFC;
        margin-bottom: 20px;
        border: 1px solid rgba(78, 205, 196, 0.25);
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }}

    .chatbot-header-left {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .bot-avatar-badge {{
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4ECDC4 0%, #2563EB 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.35);
    }}

    .chatbot-title {{
        font-family: var(--font-heading);
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
        background: linear-gradient(90deg, #FFFFFF 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .chatbot-subtitle {{
        color: #94A3B8;
        font-size: 0.88rem;
        margin-top: 4px;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22C55E;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}

    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 8px #22C55E;
        animation: pulse-dot 2s infinite;
    }}

    @keyframes pulse-dot {{
        0% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(1.2); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}

    /* Welcome / Empty state card */
    .welcome-container {{
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        margin: 12px 0 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    }}

    .welcome-heading {{
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }}

    .welcome-subtext {{
        color: #64748B;
        font-size: 0.95rem;
        max-width: 600px;
        margin: 0 auto 18px auto;
        line-height: 1.5;
    }}

    .chips-label {{
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 10px;
    }}

    /* Quick prompt button chips */
    div[data-testid="stButton"] > button {{
        border-radius: 9999px !important;
        font-size: 0.84rem !important;
        padding: 6px 14px !important;
        font-weight: 500 !important;
        border: 1px solid #CBD5E1 !important;
        background: #FFFFFF !important;
        color: #1E293B !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }}

    div[data-testid="stButton"] > button:hover {{
        border-color: #4ECDC4 !important;
        background: #F0FDFA !important;
        color: #0D9488 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(78, 205, 196, 0.2) !important;
    }}

    /* Streamlit chat message containers with guaranteed contrast */
    div[data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid #E2E8F0;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        transition: border-color 0.2s;
    }}

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] span {{
        color: #0F172A !important;
        font-size: 1rem;
        line-height: 1.6;
    }}

    .assistant-answer {{
        font-size: 1.05rem;
        line-height: 1.65;
        color: #0F172A !important;
        font-weight: 500;
        margin-bottom: 6px;
    }}

    div[data-testid="stChatMessage"]:hover {{
        border-color: #CBD5E1;
    }}

    /* Chat bubble response metadata badge card */
    .metadata-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 8px;
        background: #F1F5F9;
        margin-top: 10px;
        border-left: 3px solid #4ECDC4;
        font-size: 0.82rem;
    }}

    .metadata-bar.fallback {{
        background: #FFFBEB;
        border-left: 3px solid #F59E0B;
    }}

    .badge-category {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #E2E8F0;
        color: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 600;
    }}

    .metric-badge {{
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }}

    .metric-badge-high {{
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }}

    .metric-badge-mid {{
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }}

    .metric-badge-low {{
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }}

    /* Sidebar custom styles */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid #334155;
    }}

    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label {{
        color: #F1F5F9 !important;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: #334155;
        margin: 14px 0;
    }}

    .sidebar-brand {{
        padding: 8px 0 16px 0;
        text-align: center;
        border-bottom: 1px solid #334155;
        margin-bottom: 16px;
    }}

    .sidebar-brand h2 {{
        margin: 0;
        color: #F8FAFC;
        font-size: 1.5rem;
    }}

    .sidebar-brand span {{
        color: #4ECDC4;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .stat-card-sidebar {{
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }}

    .stat-num {{
        font-family: var(--font-heading);
        font-size: 1.6rem;
        font-weight: 700;
        color: #4ECDC4;
    }}

    .stat-lbl {{
        font-size: 0.74rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 4px;
        margin-bottom: 16px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }}

    /* Fullscreen button hidden on charts for sleek look */
    button[title="View fullscreen"] {{
        visibility: hidden;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def init_session_state() -> None:
    """Initializes session state variables for chat history and retrieval metrics."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    if "successful_retrievals" not in st.session_state:
        st.session_state.successful_retrievals = 0

    if "failed_retrievals" not in st.session_state:
        st.session_state.failed_retrievals = 0

    if "scores_history" not in st.session_state:
        st.session_state.scores_history = []

    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


def record_query_stats(found: bool, score: float) -> None:
    """Updates analytical metrics in session state after a query is processed."""
    st.session_state.total_queries += 1
    if found:
        st.session_state.successful_retrievals += 1
    else:
        st.session_state.failed_retrievals += 1
    st.session_state.scores_history.append(score)


def clear_chat_history() -> None:
    """Resets chat conversation and analytics while retaining model state."""
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def format_score_badge(score: float, threshold: float) -> str:
    """Returns an HTML badge styled according to similarity score magnitude."""
    percentage = int(round(score * 100))
    if score >= 0.70:
        css_class = "metric-badge metric-badge-high"
        icon = "🎯 High Confidence"
    elif score >= threshold:
        css_class = "metric-badge metric-badge-mid"
        icon = "✅ Accepted Match"
    else:
        css_class = "metric-badge metric-badge-low"
        icon = "⚠️ Low Confidence"

    return f'<span class="{css_class}">{icon} • {percentage}% ({score:.2f})</span>'


def build_category_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Generates an aesthetic Plotly donut chart of FAQs grouped by category."""
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    colors = [
        "#4ECDC4", "#38B2AC", "#2DD4BF", "#0D9488", "#14B8A6",
        "#06B6D4", "#0EA5E9", "#3B82F6", "#6366F1", "#8B5CF6"
    ]

    fig = go.Figure(data=[go.Pie(
        labels=counts["Category"],
        values=counts["Count"],
        hole=0.58,
        marker=dict(colors=colors[:len(counts)], line=dict(color="#0F172A", width=2)),
        textinfo="label+value",
        textposition="outside",
        hoverinfo="label+percent+value",
    )])

    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#334155", size=12),
    )
    return fig


def build_candidate_scores_chart(top_matches: List[Dict[str, Any]], threshold: float) -> go.Figure:
    """Builds a horizontal bar chart displaying top candidate similarity scores."""
    if not top_matches:
        return go.Figure()

    questions = [m["question"][:38] + "..." if len(m["question"]) > 38 else m["question"] for m in top_matches]
    scores = [m["score"] for m in top_matches]

    # Reverse order so best match is at the top of horizontal bar
    questions = questions[::-1]
    scores = scores[::-1]

    colors = ["#4ECDC4" if s >= threshold else "#F59E0B" for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=questions,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#0F172A", width=1)),
        text=[f"{s:.2f} ({int(round(s*100))}%)" for s in scores],
        textposition="outside",
    ))

    # Add threshold vertical dashed line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis=dict(range=[0, 1.1], title="Cosine Similarity Score", gridcolor="#E2E8F0"),
        yaxis=dict(title="Candidate FAQ"),
        margin=dict(t=30, b=30, l=10, r=40),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=11),
    )
    return fig


def build_session_trend_chart(scores_history: List[float], threshold: float) -> go.Figure:
    """Builds a clean line chart showing similarity scores across the session."""
    if not scores_history:
        return go.Figure()

    scores_df = pd.DataFrame({
        "Query Index": list(range(1, len(scores_history) + 1)),
        "Similarity Score": scores_history
    })

    fig = px.line(
        scores_df,
        x="Query Index",
        y="Similarity Score",
        markers=True,
        title="Similarity Scores Across Session Queries",
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="bottom right",
    )
    fig.update_traces(
        line=dict(color="#4ECDC4", width=3),
        marker=dict(size=8, color="#0F172A", line=dict(color="#4ECDC4", width=2))
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], title="Cosine Similarity", gridcolor="#E2E8F0"),
        xaxis=dict(title="Query #", dtick=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        font=dict(family="Inter", size=12),
        margin=dict(t=40, b=30, l=40, r=20),
    )
    return fig
