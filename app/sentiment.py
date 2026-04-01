from functools import lru_cache
from typing import Any

from transformers import pipeline


SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMOTION_LABELS = ["joy", "sadness", "anxiety", "anger", "excitement", "stress"]


def _normalize_sentiment_label(label: str) -> str:
    normalized = label.strip().lower()

    # Handle cases where a model returns generic labels (LABEL_0..2).
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }
    return mapping.get(normalized, normalized)


@lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """Load and cache the sentiment analysis pipeline locally."""
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    """Load and cache the zero-shot classification pipeline locally."""
    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Return normalized sentiment label and confidence score for text."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    sentiment_pipe = get_sentiment_pipeline()
    result = sentiment_pipe(text, truncation=True)[0]

    return {
        "label": _normalize_sentiment_label(result["label"]),
        "score": float(result["score"]),
    }


def extract_top_emotions(text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k emotions and scores using zero-shot classification."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    zero_shot_pipe = get_zero_shot_pipeline()
    result = zero_shot_pipe(
        text,
        candidate_labels=EMOTION_LABELS,
        multi_label=True,
    )

    labels = result.get("labels", [])
    scores = result.get("scores", [])

    emotions = [
        {"emotion": label, "score": float(score)}
        for label, score in zip(labels, scores)
    ]
    return emotions[:top_k]


def analyze_entry(text: str) -> dict[str, Any]:
    """Run full emotional analysis for a journal entry."""
    sentiment = analyze_sentiment(text)
    emotions = extract_top_emotions(text, top_k=3)

    return {
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "emotions": emotions,
    }
