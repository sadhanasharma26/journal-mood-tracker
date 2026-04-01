from functools import lru_cache
import os
import platform
from typing import Any


SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMOTION_LABELS = ["joy", "sadness", "anxiety", "anger", "excitement", "stress"]
POSITIVE_WORDS = {
    "good", "great", "happy", "calm", "focused", "productive", "proud", "grateful",
    "joy", "love", "optimistic", "excited", "progress", "better", "peaceful",
}
NEGATIVE_WORDS = {
    "bad", "sad", "angry", "anxious", "stress", "stressed", "tired", "upset",
    "overwhelmed", "worse", "panic", "irritated", "frustrated", "heavy", "low",
}


def _is_light_mode_enabled() -> bool:
    value = os.getenv("JMT_LIGHT_MODE", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


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

    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    """Load and cache the zero-shot classification pipeline locally."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Import lazily so API startup does not depend on ML runtime initialization.
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)


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
