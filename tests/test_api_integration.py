from __future__ import annotations

import math
import os
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient
from app.main import app, require_local_or_token
import app.main as main_module


def _unit_embedding() -> np.ndarray:
    vec = np.ones(384, dtype=np.float32)
    vec /= math.sqrt(384)
    return vec


def test_entries_search_and_insights_endpoints(test_db, monkeypatch):
    _engine, _TestingSessionLocal, override_get_db = test_db

    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(
        main_module,
        "analyze_entry",
        lambda text: {
            "sentiment_label": "positive",
            "sentiment_score": 0.91,
            "emotions": [
                {"emotion": "joy", "score": 0.88},
                {"emotion": "excitement", "score": 0.41},
                {"emotion": "stress", "score": 0.13},
            ],
        },
    )
    monkeypatch.setattr(main_module, "generate_embedding", lambda text: _unit_embedding())
    monkeypatch.setattr(main_module, "generate_insight", lambda entries: "Mocked insight")
    monkeypatch.setattr(
        main_module,
        "semantic_search",
        lambda query, index, id_map, top_k=5: [
            {"entry_id": id_map[0], "score": 0.99}
        ]
        if id_map
        else [],
    )

    app.dependency_overrides[main_module.get_db] = override_get_db

    try:
        with TestClient(app) as client:
            day_a = date.today().strftime("%Y-%m-%d")
            day_b = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

            response = client.post(
                "/entries",
                json={"date": day_a, "text": "Great day with deep focus."},
            )
            assert response.status_code == 200

            response = client.post(
                "/entries",
                json={"date": day_b, "text": "Calm and optimistic overall."},
            )
            assert response.status_code == 200

            response = client.get("/entries")
            assert response.status_code == 200
            entries = response.json()
            assert len(entries) == 2

            response = client.get(f"/entries/{day_a}")
            assert response.status_code == 200
            assert response.json()["date"] == day_a

            response = client.get("/search", params={"q": "optimistic", "top_k": 5})
            assert response.status_code == 200
            results = response.json()["results"]
            assert len(results) >= 1
            assert "semantic_score" in results[0]

            response = client.get("/insights/weekly")
            assert response.status_code == 200
            assert response.json()["insight"] == "Mocked insight"

            response = client.get("/insights/monthly")
            assert response.status_code == 200
            assert response.json()["insight"] == "Mocked insight"
    finally:
        app.dependency_overrides.clear()


def _make_request(host: str, token_header: str | None = None) -> MagicMock:
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = host
    headers = {}
    if token_header is not None:
        headers["x-api-token"] = token_header
    req.headers = headers
    return req


def test_remote_request_without_token_returns_403(monkeypatch):
    monkeypatch.delenv("LOCAL_API_TOKEN", raising=False)
    from fastapi import HTTPException
    req = _make_request("203.0.113.5")
    try:
        require_local_or_token(req)
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 403


def test_remote_request_with_valid_token_passes(monkeypatch):
    monkeypatch.setenv("LOCAL_API_TOKEN", "supersecrettoken123")
    req = _make_request("203.0.113.5", token_header="supersecrettoken123")
    require_local_or_token(req)


def test_local_request_always_passes(monkeypatch):
    monkeypatch.delenv("LOCAL_API_TOKEN", raising=False)
    req = _make_request("127.0.0.1")
    require_local_or_token(req)


def test_put_entry_updates_record(test_db, monkeypatch):
    _engine, _TestingSessionLocal, override_get_db = test_db

    monkeypatch.setattr(main_module, "init_db", lambda: None)
    monkeypatch.setattr(
        main_module,
        "analyze_entry",
        lambda text: {
            "sentiment_label": "negative",
            "sentiment_score": 0.75,
            "emotions": [{"emotion": "sadness", "score": 0.8}],
        },
    )
    monkeypatch.setattr(main_module, "generate_embedding", lambda text: _unit_embedding())
    app.dependency_overrides[main_module.get_db] = override_get_db

    try:
        with TestClient(app) as client:
            day = date.today().strftime("%Y-%m-%d")
            monkeypatch.setattr(
                main_module,
                "analyze_entry",
                lambda text: {
                    "sentiment_label": "positive",
                    "sentiment_score": 0.9,
                    "emotions": [{"emotion": "joy", "score": 0.9}],
                },
            )
            client.post("/entries", json={"date": day, "text": "Original text."})

            monkeypatch.setattr(
                main_module,
                "analyze_entry",
                lambda text: {
                    "sentiment_label": "negative",
                    "sentiment_score": 0.75,
                    "emotions": [{"emotion": "sadness", "score": 0.8}],
                },
            )
            resp = client.put(f"/entries/{day}", json={"text": "Updated text, feeling sad."})
            assert resp.status_code == 200
            data = resp.json()
            assert data["raw_text"] == "Updated text, feeling sad."
            assert data["sentiment_label"] == "negative"
    finally:
        app.dependency_overrides.clear()
