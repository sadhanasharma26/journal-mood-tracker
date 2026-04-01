import json
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EMOTIONS_ORDER = ["joy", "sadness", "anxiety", "anger", "excitement", "stress"]


st.set_page_config(page_title="Journal Mood Tracker", page_icon="📝", layout="wide")
st.title("Privacy-First Journal + Mood Tracker")
st.caption("Everything runs locally. No data leaves your machine.")


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


entries = _fetch_entries()

home_tab, timeline_tab, heatmap_tab, weekly_tab, search_tab = st.tabs(
    ["Home", "Timeline", "Emotion Heatmap", "Weekly Insight", "Search"]
)

with home_tab:
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

with timeline_tab:
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
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

with heatmap_tab:
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
            color_continuous_scale="Sunset",
            title="Emotion Intensity Heatmap",
            category_orders={"day": DAYS_ORDER, "emotion": EMOTIONS_ORDER},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(_stress_pattern_note(heatmap_df))

with weekly_tab:
    st.subheader("Weekly Insight")
    if st.button("Generate Weekly Insight"):
        response = _safe_get("/insights/weekly")
        if response is None:
            st.stop()

        if response.status_code == 200:
            insight = response.json().get("insight", "No insight returned.")
            st.info(insight)
        else:
            st.error(f"Failed to fetch weekly insight: {response.text}")

with search_tab:
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
                        st.markdown(
                            f"**{item['date']}** | sentiment: `{item['sentiment_label']}` | "
                            f"score: `{item.get('semantic_score', 0):.3f}`"
                        )
                        st.write(item["raw_text"])
                        st.divider()
            else:
                st.error(f"Search failed: {response.text}")
