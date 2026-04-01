import math
from datetime import date, timedelta

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.main import app
import app.main as main_module


def _unit_embedding() -> np.ndarray:
    vec = np.ones(384, dtype=np.float32)
    vec /= math.sqrt(384)
    return vec


def test_entries_search_and_insights_endpoints(tmp_path, monkeypatch):
    test_db_path = tmp_path / "api_integration.db"
    test_engine = create_engine(
        f"sqlite:///{test_db_path}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

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
