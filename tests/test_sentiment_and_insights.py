from app import insights, sentiment


def test_analyze_entry_with_mocked_pipelines(monkeypatch):
    def fake_sentiment_pipeline(text, truncation=True):
        assert truncation is True
        return [{"label": "positive", "score": 0.91}]

    def fake_zero_shot_pipeline(text, candidate_labels, multi_label=True):
        assert multi_label is True
        assert candidate_labels == sentiment.EMOTION_LABELS
        return {
            "labels": ["joy", "excitement", "stress", "sadness"],
            "scores": [0.82, 0.76, 0.21, 0.1],
        }

    monkeypatch.setattr(sentiment, "get_sentiment_pipeline", lambda: fake_sentiment_pipeline)
    monkeypatch.setattr(sentiment, "get_zero_shot_pipeline", lambda: fake_zero_shot_pipeline)

    result = sentiment.analyze_entry("Today felt productive and exciting.")

    assert result["sentiment_label"] == "positive"
    assert result["sentiment_score"] == 0.91
    assert len(result["emotions"]) == 3
    assert result["emotions"][0]["emotion"] == "joy"


def test_generate_insight_mocks_ollama_chat(monkeypatch):
    class FakeClient:
        def chat(self, model, messages):
            assert model == "llama3"
            assert len(messages) == 2
            assert "journal entries" in messages[1]["content"].lower()
            return {"message": {"content": "You showed a steadier mood this week."}}

    monkeypatch.setattr(insights, "Client", FakeClient)

    entries = [
        {
            "date": "2026-03-30",
            "raw_text": "Had a stressful Monday but felt better after exercise.",
            "sentiment_label": "neutral",
            "sentiment_score": 0.62,
            "emotions": [
                {"emotion": "stress", "score": 0.77},
                {"emotion": "joy", "score": 0.31},
            ],
        }
    ]

    output = insights.generate_insight(entries, model="llama3")
    assert "steadier mood" in output.lower()


def test_generate_insight_empty_entries():
    output = insights.generate_insight([])
    assert "not enough recent entries" in output.lower()


def test_generate_insight_handles_ollama_unavailable(monkeypatch):
    class FailingClient:
        def chat(self, model, messages):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(insights, "Client", FailingClient)

    entries = [
        {
            "date": "2026-03-30",
            "raw_text": "Work was stressful.",
            "sentiment_label": "negative",
            "sentiment_score": 0.8,
            "emotions": [{"emotion": "stress", "score": 0.9}],
        }
    ]

    output = insights.generate_insight(entries, model="llama3")
    assert "ollama" in output.lower()
    assert "temporarily unavailable" in output.lower()
