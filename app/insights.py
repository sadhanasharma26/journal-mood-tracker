import json
import logging
import os
from typing import Any

from ollama import Client


DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
logger = logging.getLogger(__name__)


def _format_entries(entries: list[dict[str, Any]]) -> str:
    """Convert entries into compact JSON lines for prompt grounding."""
    formatted: list[dict[str, Any]] = []

    for item in entries:
        formatted.append(
            {
                "date": item.get("date"),
                "summary": item.get("summary") or item.get("raw_text", "")[:220],
                "sentiment": item.get("sentiment_label"),
                "sentiment_score": item.get("sentiment_score"),
                "emotions": item.get("emotions"),
            }
        )

    return json.dumps(formatted, ensure_ascii=True, indent=2)


def generate_insight(entries: list[dict[str, Any]], model: str = DEFAULT_OLLAMA_MODEL) -> str:
    """
    Generate an empathetic insight summary from journal entries using local Ollama.

    Expected entry fields include: date, raw_text/summary, sentiment_label,
    sentiment_score, and emotions.
    """
    if not entries:
        return (
            "Not enough recent entries yet. Add a few journal notes this week to "
            "generate personalized insights."
        )

    system_prompt = (
        "You are a compassionate mental wellness assistant. "
        "Analyze emotional patterns in journal data and provide concise, practical guidance. "
        "Do not diagnose medical conditions."
    )

    user_prompt = (
        "Here are journal entries from the past period with their emotional analysis:\n"
        "[date, summary, sentiment, emotions]\n\n"
        f"{_format_entries(entries)}\n\n"
        "Identify patterns, recurring stressors, mood trends, and give 3 actionable insights. "
        "Be empathetic and concise.\n"
        "Response format:\n"
        "1) Mood trend (2-3 sentences)\n"
        "2) Recurring stressors/triggers (bullet points)\n"
        "3) Three actionable insights (numbered list)"
    )

    try:
        client = Client()
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()
    except Exception:
        logger.exception(
            "Insight generation failed",
            extra={"model": model, "entries_count": len(entries)},
        )
        # Keep the API responsive when Ollama is unavailable or model is missing.
        return (
            "Insight generation is temporarily unavailable because Ollama could not be reached. "
            "Start Ollama (`ollama serve`) and ensure the model is available "
            f"(`ollama pull {model}`)."
        )
