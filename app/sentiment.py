from functools import lru_cache
import os
from typing import Any

from app.config import settings
from app.utils import _is_light_mode_enabled


SENTIMENT_MODEL = settings.SENTIMENT_MODEL
ZERO_SHOT_MODEL = settings.ZERO_SHOT_MODEL
EMOTION_LABELS = settings.EMOTION_LABELS
POSITIVE_WORDS = settings.POSITIVE_WORDS
NEGATIVE_WORDS = settings.NEGATIVE_WORDS


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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Import lazily so API startup does not depend on ML runtime initialization.
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=settings.SENTIMENT_MODEL)


@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    """Load and cache the zero-shot classification pipeline locally."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Import lazily so API startup does not depend on ML runtime initialization.
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=settings.ZERO_SHOT_MODEL)


def analyze_sentiment(text: str) -> dict[str, Any]:
    """Return normalized sentiment label and confidence score for text."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty.")

    if _is_light_mode_enabled():
        tokens = [t.strip(".,!?;:\"'()[]{}").lower() for t in text.split()]
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        if pos > neg:
            return {"label": "positive", "score": min(0.99, 0.6 + 0.08 * (pos - neg))}
        if neg > pos:
            return {"label": "negative", "score": min(0.99, 0.6 + 0.08 * (neg - pos))}
        return {"label": "neutral", "score": 0.55}

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

    if _is_light_mode_enabled():
        lowered = text.lower()
        scores = {
            "joy": 0.2,
            "sadness": 0.2,
            "anxiety": 0.2,
            "anger": 0.2,
            "excitement": 0.2,
            "stress": 0.2,
        }
        keyword_map = {
            "joy": ["happy", "grateful", "joy", "smile", "calm"],
            "sadness": ["sad", "down", "lonely", "low"],
            "anxiety": ["anxious", "worry", "panic", "nervous"],
            "anger": ["angry", "frustrated", "irritated", "mad"],
            "excitement": ["excited", "thrilled", "energized", "progress"],
            "stress": ["stress", "overwhelmed", "deadline", "pressure", "heavy"],
        }
        for emotion, keywords in keyword_map.items():
            hits = sum(1 for k in keywords if k in lowered)
            scores[emotion] += 0.2 * hits

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [{"emotion": e, "score": float(min(0.99, s))} for e, s in ranked[:top_k]]

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
