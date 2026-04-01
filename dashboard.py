import json
from datetime import date
from html import escape

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EMOTIONS_ORDER = ["joy", "sadness", "anxiety", "anger", "excitement", "stress"]


st.set_page_config(page_title="Journal Mood Tracker", page_icon="Journal", layout="wide")

st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

            :root {
                --bg-a: #fdf6ee;
                --bg-b: #edf9f4;
                --panel: #ffffffcc;
                --ink: #111827;
                --muted: #344054;
                --brand: #f36f45;
                --brand-2: #ff9b54;
                --brand-soft: #ffe7dc;
                --accent: #169d7f;
                --accent-soft: #dff6ef;
                --line: #dce3eb;
            }

            .stApp {
                background:
                    radial-gradient(circle at 10% 0%, #ffe9d6 0%, transparent 35%),
                    radial-gradient(circle at 100% 10%, #dbf6ef 0%, transparent 36%),
                    linear-gradient(145deg, var(--bg-a), var(--bg-b));
            }

            .main .block-container {
                max-width: 1160px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.9rem;
            }

            h1, h2, h3, p, label, span, div {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
            }

            .stCaption, .stCaption p {
                color: #344054 !important;
            }

            .stMarkdown p {
                color: var(--ink);
            }

            .stTextInput label, .stTextArea label, .stSlider label {
                color: #1f2937 !important;
                font-weight: 600;
            }

            .stSlider div, .stSlider span {
                color: #334155 !important;
            }

            .stAlert {
                color: #111827;
            }

            .hero {
                position: relative;
                overflow: hidden;
                background: linear-gradient(130deg, #ffffffde, #fff6f0cc);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 1.1rem 1.2rem;
                margin-bottom: 1rem;
                animation: rise 420ms ease-out;
            }

            .hero::after {
                content: "";
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--brand), var(--brand-2), var(--accent));
                opacity: .9;
            }

            .hero-title {
                font-size: clamp(1.6rem, 3.4vw, 2.8rem);
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.02em;
            }

            .hero-sub {
                margin-top: .4rem;
                color: var(--muted);
                font-size: 1.05rem;
            }

            .pill {
                display: inline-block;
                padding: .3rem .65rem;
                border-radius: 999px;
                font-size: .78rem;
                font-weight: 700;
                background: var(--brand-soft);
                color: #b12f1a;
                margin-bottom: .55rem;
            }

            .metric-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: .9rem 1rem;
                transition: transform .16s ease, box-shadow .16s ease;
            }

            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 26px #25344f1a;
            }

            .metric-label {
                color: var(--muted);
                font-weight: 600;
                font-size: .86rem;
            }

            .metric-value {
                font-weight: 800;
                font-size: 1.35rem;
                margin-top: .2rem;
            }

            .section-card {
                background: #fffffff2;
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 1rem 1rem 1.1rem 1rem;
                margin-top: 0.35rem;
                animation: rise 500ms ease-out;
            }

            .stTabs {
                margin-top: 0.6rem;
            }

            [data-baseweb="tab-list"] {
                display: flex;
                flex-wrap: wrap;
                gap: .55rem;
                margin-bottom: .65rem;
                padding-bottom: .25rem;
            }

            [data-baseweb="tab"] {
                border-radius: 999px;
                border: 1px solid var(--line);
                background: #ffffffdd;
                color: #1f2937 !important;
                padding: .5rem .95rem;
                min-height: 40px;
                transition: all .16s ease;
            }

            [data-baseweb="tab"]:hover {
                border-color: #f1b8a8;
                background: #fff6f2;
            }

            [aria-selected="true"] {
                color: #ffffff !important;
                background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important;
                border-color: transparent !important;
                box-shadow: 0 7px 18px #f36f453d;
            }

            [data-baseweb="tab-highlight"] {
                background: transparent !important;
            }

            .stButton > button {
                border-radius: 12px;
                border: 1px solid #efb8a7;
                background: linear-gradient(90deg, var(--brand), var(--brand-2));
                color: #fff;
                font-weight: 700;
                letter-spacing: .01em;
                min-height: 42px;
                padding: .45rem 1.05rem;
                transition: transform .16s ease, box-shadow .16s ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 24px #f36f4535;
            }

            .stTextInput input, .stTextArea textarea {
                border-radius: 12px !important;
                border: 1px solid #d4dbe6 !important;
                background: #ffffffee !important;
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                font-weight: 500;
            }

            .stTextInput input::placeholder, .stTextArea textarea::placeholder {
                color: #64748b !important;
                opacity: 1 !important;
            }

            .stTextInput input:focus, .stTextArea textarea:focus {
                border: 1px solid #2b5f7f !important;
                box-shadow: 0 0 0 2px #c8dff0 !important;
            }

            .stSlider [role="slider"] {
                background: var(--accent) !important;
                border-color: var(--accent) !important;
            }

            .stSlider [data-testid="stTickBar"] * {
                color: #334155 !important;
            }

            .search-hit {
                border-left: 5px solid var(--accent);
                background: linear-gradient(135deg, #fffffff2, var(--accent-soft));
                border-radius: 12px;
                padding: .75rem .9rem;
                margin-bottom: .7rem;
                transition: transform .14s ease;
            }

            .search-hit:hover {
                transform: translateX(2px);
            }

            .insight-card {
                border: 1px solid #ffd3c7;
                background: linear-gradient(135deg, #fff6f1, #ffffff);
                border-radius: 14px;
                padding: .85rem 1rem;
                line-height: 1.45;
            }

            .mono {
                font-family: 'IBM Plex Mono', monospace;
            }

            @keyframes rise {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }

            @media (max-width: 760px) {
                .main .block-container { padding-top: 1rem; }
                .hero { padding: .9rem; }
                .hero-sub { font-size: .95rem; }
                [data-baseweb="tab"] { min-height: 38px; padding: .42rem .8rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
)


def _safe_get(path: str, params: dict | None = None) -> requests.Response | None:
    try:
        return requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return None


def _safe_post(path: str, payload: dict) -> requests.Response | None:
    try:
        return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_BASE_URL}: {exc}")
        return None


def _fetch_entries() -> list[dict]:
    response = _safe_get("/entries")
    if response is None:
        return []
    if response.status_code != 200:
        st.error(f"Failed to load entries: {response.text}")
        return []
    return response.json()


def _build_emotion_heatmap_df(entries: list[dict]) -> pd.DataFrame:
    rows = []

    for entry in entries:
        day_name = pd.to_datetime(entry["date"]).day_name()[:3]
        try:
            emotions = entry.get("emotions", [])
            if isinstance(emotions, str):
                emotions = json.loads(emotions)
        except json.JSONDecodeError:
            emotions = []

        for emotion in emotions:
            rows.append(
                {
                    "day": day_name,
                    "emotion": emotion.get("emotion"),
                    "score": float(emotion.get("score", 0.0)),
                }
            )

    if not rows:
        # Create a zero-filled frame so the heatmap structure is consistent.
        rows = [
            {"day": day, "emotion": emotion, "score": 0.0}
            for day in DAYS_ORDER
            for emotion in EMOTIONS_ORDER
        ]

    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(
        index="emotion",
        columns="day",
        values="score",
        aggfunc="sum",
        fill_value=0.0,
    )

    for day in DAYS_ORDER:
        if day not in pivot.columns:
            pivot[day] = 0.0

    for emotion in EMOTIONS_ORDER:
        if emotion not in pivot.index:
            pivot.loc[emotion] = [0.0] * len(pivot.columns)

    pivot = pivot.loc[EMOTIONS_ORDER, DAYS_ORDER]
    return pivot.reset_index().melt(id_vars="emotion", var_name="day", value_name="score")


def _stress_pattern_note(heatmap_df: pd.DataFrame) -> str:
    stress_rows = heatmap_df[heatmap_df["emotion"] == "stress"]
    if stress_rows.empty:
        return "No stress pattern detected yet."

    peak_row = stress_rows.sort_values("score", ascending=False).iloc[0]
    if peak_row["score"] <= 0:
        return "No stress pattern detected yet."

    return f"Pattern note: stress appears highest on {peak_row['day']}."


def _sentiment_snapshot(entries: list[dict]) -> tuple[str, float, str]:
    if not entries:
        return "No data", 0.0, "neutral"

    frame = pd.DataFrame(entries)
    mean_score = float(frame["sentiment_score"].mean())

    if mean_score >= 0.67:
        return "Upward", mean_score, "positive"
    if mean_score <= 0.38:
        return "Low", mean_score, "negative"
    return "Steady", mean_score, "neutral"


def _as_safe_html(text: str) -> str:
    return escape(text).replace("\n", "<br>")


entries = _fetch_entries()
trend_label, avg_sentiment, trend_kind = _sentiment_snapshot(entries)

trend_color = {
    "positive": "#0f9f80",
    "neutral": "#d99310",
    "negative": "#d14c4c",
}.get(trend_kind, "#5f6777")

st.markdown(
    """
    <section class="hero">
      <div class="pill">Local AI + Private by Design</div>
      <h1 class="hero-title">Privacy-First Journal + Mood Tracker</h1>
      <p class="hero-sub">Everything runs locally. No data leaves your machine.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Total Entries</div>
          <div class="metric-value mono">{len(entries)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Average Sentiment</div>
          <div class="metric-value mono">{avg_sentiment:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">Mood Direction</div>
          <div class="metric-value" style="color:{trend_color};">{trend_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

home_tab, timeline_tab, heatmap_tab, weekly_tab, search_tab = st.tabs(
    ["Home", "Timeline", "Emotion Heatmap", "Weekly Insight", "Search"]
)

with home_tab:
    st.markdown('<section class="section-card">', unsafe_allow_html=True)
    st.subheader("New Journal Entry")
    default_date = date.today().strftime("%Y-%m-%d")
    selected_date = st.text_input("Date (YYYY-MM-DD)", value=default_date)
    text = st.text_area("How was your day?", height=180)

    if st.button("Save Entry", type="primary"):
        if not text.strip():
            st.warning("Please write something before submitting.")
        else:
            payload = {"date": selected_date, "text": text.strip()}
            response = _safe_post("/entries", payload)
            if response is None:
                st.stop()

            if response.status_code in (200, 201):
                st.success("Entry saved and analyzed locally.")
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
            else:
                st.error(f"Failed to save entry: {response.text}")
    st.markdown("</section>", unsafe_allow_html=True)

with timeline_tab:
    st.markdown('<section class="section-card">', unsafe_allow_html=True)
    st.subheader("Sentiment Timeline")
    if not entries:
        st.info("No entries yet. Add a journal entry on the Home tab.")
    else:
        df = pd.DataFrame(entries)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        fig = px.line(
            df,
            x="date",
            y="sentiment_score",
            markers=True,
            title="Sentiment Score Over Time",
            labels={"sentiment_score": "Sentiment Score", "date": "Date"},
            hover_data=["sentiment_label"],
        )
        fig.update_traces(line=dict(color="#ff5a3d", width=3), marker=dict(size=8))
        fig.update_layout(
            yaxis_range=[0, 1],
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#1f2937", size=14),
            xaxis=dict(title_font=dict(color="#1f2937"), tickfont=dict(color="#334155")),
            yaxis=dict(title_font=dict(color="#1f2937"), tickfont=dict(color="#334155")),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</section>", unsafe_allow_html=True)

with heatmap_tab:
    st.markdown('<section class="section-card">', unsafe_allow_html=True)
    st.subheader("Emotion Frequency by Day of Week")
    if not entries:
        st.info("No entries yet. Add data to view emotion patterns.")
    else:
        heatmap_df = _build_emotion_heatmap_df(entries)
        fig = px.density_heatmap(
            heatmap_df,
            x="day",
            y="emotion",
            z="score",
            color_continuous_scale=["#fff2e7", "#ffb595", "#ff6d49", "#c63a1e"],
            title="Emotion Intensity Heatmap",
            category_orders={"day": DAYS_ORDER, "emotion": EMOTIONS_ORDER},
        )
        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#1f2937", size=14),
            xaxis=dict(title_font=dict(color="#1f2937"), tickfont=dict(color="#334155")),
            yaxis=dict(title_font=dict(color="#1f2937"), tickfont=dict(color="#334155")),
            coloraxis_colorbar=dict(
                title_font=dict(color="#1f2937"),
                tickfont=dict(color="#334155"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(_stress_pattern_note(heatmap_df))
    st.markdown("</section>", unsafe_allow_html=True)

with weekly_tab:
    st.markdown('<section class="section-card">', unsafe_allow_html=True)
    st.subheader("Weekly Insight")
    if st.button("Generate Weekly Insight"):
        response = _safe_get("/insights/weekly")
        if response is None:
            st.stop()

        if response.status_code == 200:
            insight = response.json().get("insight", "No insight returned.")
            st.markdown(
                f"<div class='insight-card'>{_as_safe_html(insight)}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(f"Failed to fetch weekly insight: {response.text}")
    st.markdown("</section>", unsafe_allow_html=True)

with search_tab:
    st.markdown('<section class="section-card">', unsafe_allow_html=True)
    st.subheader("Semantic Search")
    query = st.text_input("Search entries by meaning", value="")
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            response = _safe_get("/search", params={"q": query.strip(), "top_k": top_k})
            if response is None:
                st.stop()

            if response.status_code == 200:
                results = response.json().get("results", [])
                if not results:
                    st.info("No semantically similar entries found.")
                else:
                    for item in results:
                        safe_date = _as_safe_html(str(item["date"]))
                        safe_sentiment = _as_safe_html(str(item["sentiment_label"]))
                        safe_score = f"{item.get('semantic_score', 0):.3f}"
                        safe_text = _as_safe_html(str(item["raw_text"]))
                        st.markdown(
                            f"""
                            <div class="search-hit">
                              <strong>{safe_date}</strong> | sentiment:
                              <span class="mono">{safe_sentiment}</span> |
                              similarity: <span class="mono">{safe_score}</span>
                              <div style="margin-top:.35rem;">{safe_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.error(f"Search failed: {response.text}")
    st.markdown("</section>", unsafe_allow_html=True)
