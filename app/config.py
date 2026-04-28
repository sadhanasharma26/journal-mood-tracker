import os


class Settings:
    SENTIMENT_MODEL: str = os.getenv(
        "JMT_SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    ZERO_SHOT_MODEL: str = os.getenv(
        "JMT_ZERO_SHOT_MODEL", "facebook/bart-large-mnli"
    )
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "JMT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    DEFAULT_OLLAMA_MODEL: str = os.getenv("JMT_OLLAMA_MODEL") or os.getenv(
        "OLLAMA_MODEL", "llama3"
    )
    EMOTION_LABELS: list[str] = [
        "joy", "sadness", "anxiety", "anger", "excitement", "stress"
    ]
    POSITIVE_WORDS: set[str] = {
        "good", "great", "happy", "calm", "focused", "productive", "proud", "grateful",
        "joy", "love", "optimistic", "excited", "progress", "better", "peaceful",
    }
    NEGATIVE_WORDS: set[str] = {
        "bad", "sad", "angry", "anxious", "stress", "stressed", "tired", "upset",
        "overwhelmed", "worse", "panic", "irritated", "frustrated", "heavy", "low",
    }


settings = Settings()
