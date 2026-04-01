import argparse
import json
import os
from collections.abc import Generator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from threading import Lock

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import DB_PATH, SessionLocal, get_db, init_db
from app.embeddings import (
    build_faiss_index,
    generate_embedding,
    semantic_search,
    serialize_embedding,
)
from app.insights import generate_insight
from app.models import JournalEntry
from app.sentiment import analyze_entry


SAFE_MODE_ENABLED = os.getenv("SAFE_MODE", "0") == "1"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}

_faiss_cache_lock = Lock()
_faiss_cache: dict = {
    "count": -1,
    "index": None,
    "id_map": [],
}


def _is_local_request(host: str | None) -> bool:
    return bool(host and host in LOCAL_HOSTS)


def require_local_or_token(request: Request) -> None:
    client_host = request.client.host if request.client else None
    if _is_local_request(client_host):
        return

    token = os.getenv("LOCAL_API_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=403,
            detail=(
                "Remote access is disabled. Set LOCAL_API_TOKEN to enable "
                "non-local requests."
            ),
        )

    header_token = request.headers.get("x-api-token", "")
    auth_header = request.headers.get("authorization", "")
    bearer_token = ""
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header.split(" ", 1)[1].strip()

    if header_token == token or bearer_token == token:
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


def _model_available(models: list[dict], target_model: str) -> bool:
    target = target_model.strip()
    target_base = target.split(":", 1)[0]

    for model in models:
        name = model.get("name", "")
        if name == target or name == target_base or name.startswith(f"{target_base}:"):
            return True
    return False


def run_safe_mode_checks() -> None:
    issues: list[str] = []
    model_name = os.getenv("OLLAMA_MODEL", "llama3")

    # Verify DB path is writable.
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DB_PATH, "ab"):
            pass
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        issues.append(f"Database is not writable: {exc}")

    # Verify Ollama service is reachable and model is available locally.
    try:
        client = OllamaClient()
        response = client.list()
        models = response.get("models", []) if isinstance(response, dict) else []
        if not _model_available(models, model_name):
            issues.append(f"Ollama model '{model_name}' is not pulled locally")
    except Exception as exc:
        issues.append(f"Ollama service not reachable: {exc}")

    if issues:
        raise RuntimeError("Safe mode startup checks failed: " + "; ".join(issues))


def _invalidate_faiss_cache() -> None:
    with _faiss_cache_lock:
        _faiss_cache["count"] = -1
        _faiss_cache["index"] = None
        _faiss_cache["id_map"] = []


def _get_or_build_faiss_index(entries: list[JournalEntry]):
    entry_count = len(entries)
    with _faiss_cache_lock:
        if _faiss_cache["index"] is None or _faiss_cache["count"] != entry_count:
            entry_dicts = [{"id": e.id, "embedding": e.embedding} for e in entries]
            index, id_map = build_faiss_index(entry_dicts)
            _faiss_cache["index"] = index
            _faiss_cache["id_map"] = id_map
            _faiss_cache["count"] = entry_count

        return _faiss_cache["index"], _faiss_cache["id_map"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if SAFE_MODE_ENABLED:
        run_safe_mode_checks()
    yield


app = FastAPI(
    title="Journal Mood Tracker API",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_local_or_token)],
)


class EntryCreate(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    text: str = Field(..., min_length=1, max_length=8000)


class EntryResponse(BaseModel):
    id: int
    date: str
    raw_text: str
    sentiment_label: str
    sentiment_score: float
    emotions: list[dict]
    created_at: datetime


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.") from exc


def _entry_to_dict(entry: JournalEntry) -> dict:
    return {
        "id": entry.id,
        "date": entry.date,
        "raw_text": entry.raw_text,
        "sentiment_label": entry.sentiment_label,
        "sentiment_score": entry.sentiment_score,
        "emotions": json.loads(entry.emotions),
        "created_at": entry.created_at,
    }


def _get_entries_in_days(db: Session, days: int) -> list[JournalEntry]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).date()
    entries = db.query(JournalEntry).order_by(JournalEntry.date.asc()).all()
    return [e for e in entries if datetime.strptime(e.date, "%Y-%m-%d").date() >= cutoff]


@app.post("/entries", response_model=EntryResponse)
def create_entry(payload: EntryCreate, db: Session = Depends(get_db)):
    _parse_date(payload.date)

    existing = db.query(JournalEntry).filter(JournalEntry.date == payload.date).first()
    if existing:
        raise HTTPException(status_code=409, detail="An entry already exists for this date.")

    analysis = analyze_entry(payload.text)
    embedding = generate_embedding(payload.text)

    entry = JournalEntry(
        date=payload.date,
        raw_text=payload.text,
        sentiment_label=analysis["sentiment_label"],
        sentiment_score=analysis["sentiment_score"],
        emotions=json.dumps(analysis["emotions"]),
        embedding=serialize_embedding(embedding),
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)
    _invalidate_faiss_cache()

    return _entry_to_dict(entry)


@app.get("/entries", response_model=list[EntryResponse])
def list_entries(db: Session = Depends(get_db)):
    entries = db.query(JournalEntry).order_by(JournalEntry.date.asc()).all()
    return [_entry_to_dict(entry) for entry in entries]


@app.get("/entries/{date}", response_model=EntryResponse)
def get_entry_by_date(date: str, db: Session = Depends(get_db)):
    _parse_date(date)

    entry = db.query(JournalEntry).filter(JournalEntry.date == date).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found for the specified date.")

    return _entry_to_dict(entry)


@app.get("/insights/weekly")
def weekly_insight(db: Session = Depends(get_db)):
    entries = _get_entries_in_days(db, 7)
    payload = [_entry_to_dict(e) for e in entries]
    return {"period": "weekly", "insight": generate_insight(payload)}


@app.get("/insights/monthly")
def monthly_insight(db: Session = Depends(get_db)):
    entries = _get_entries_in_days(db, 30)
    payload = [_entry_to_dict(e) for e in entries]
    return {"period": "monthly", "insight": generate_insight(payload)}


@app.get("/search")
def search_entries(
    q: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    entries = db.query(JournalEntry).order_by(JournalEntry.date.asc()).all()
    index, id_map = _get_or_build_faiss_index(entries)
    matches = semantic_search(q, index=index, id_map=id_map, top_k=top_k)

    if not matches:
        return {"query": q, "results": []}

    by_id = {entry.id: entry for entry in entries}
    results = []
    for match in matches:
        entry = by_id.get(match["entry_id"])
        if not entry:
            continue
        item = _entry_to_dict(entry)
        item["semantic_score"] = match["score"]
        results.append(item)

    return {"query": q, "results": results}


def seed_demo_entries(db: Session) -> int:
    if db.query(JournalEntry).count() > 0:
        return 0

    # Keep demo seeding model-free so setup is fast and stable.
    samples = [
        {
            "text": "Felt calm after a long walk and good coffee.",
            "sentiment_label": "positive",
            "sentiment_score": 0.86,
            "emotions": [
                {"emotion": "joy", "score": 0.81},
                {"emotion": "excitement", "score": 0.44},
                {"emotion": "stress", "score": 0.08},
            ],
        },
        {
            "text": "Work felt heavy today and I was anxious before meetings.",
            "sentiment_label": "negative",
            "sentiment_score": 0.84,
            "emotions": [
                {"emotion": "anxiety", "score": 0.87},
                {"emotion": "stress", "score": 0.82},
                {"emotion": "sadness", "score": 0.33},
            ],
        },
        {
            "text": "Had a productive morning and felt proud of progress.",
            "sentiment_label": "positive",
            "sentiment_score": 0.91,
            "emotions": [
                {"emotion": "joy", "score": 0.76},
                {"emotion": "excitement", "score": 0.71},
                {"emotion": "stress", "score": 0.1},
            ],
        },
        {
            "text": "Low energy all day, but a friend call helped.",
            "sentiment_label": "neutral",
            "sentiment_score": 0.61,
            "emotions": [
                {"emotion": "sadness", "score": 0.68},
                {"emotion": "joy", "score": 0.34},
                {"emotion": "stress", "score": 0.29},
            ],
        },
        {
            "text": "Excited about a new project idea and future plans.",
            "sentiment_label": "positive",
            "sentiment_score": 0.9,
            "emotions": [
                {"emotion": "excitement", "score": 0.9},
                {"emotion": "joy", "score": 0.66},
                {"emotion": "anxiety", "score": 0.22},
            ],
        },
        {
            "text": "Irritated by delays and deadlines piling up.",
            "sentiment_label": "negative",
            "sentiment_score": 0.88,
            "emotions": [
                {"emotion": "anger", "score": 0.82},
                {"emotion": "stress", "score": 0.79},
                {"emotion": "anxiety", "score": 0.36},
            ],
        },
    ]

    today = datetime.utcnow().date()
    created = 0

    for i in range(30):
        day = today - timedelta(days=29 - i)
        sample = samples[i % len(samples)]
        text = sample["text"]

        # Generate deterministic synthetic embeddings for FAISS demo search.
        seed = abs(hash(f"{day.isoformat()}::{text}")) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.normal(size=384).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-12

        entry = JournalEntry(
            date=day.strftime("%Y-%m-%d"),
            raw_text=text,
            sentiment_label=sample["sentiment_label"],
            sentiment_score=sample["sentiment_score"],
            emotions=json.dumps(sample["emotions"]),
            embedding=serialize_embedding(embedding),
        )
        db.add(entry)
        created += 1

    db.commit()
    return created


def _session_generator() -> Generator[Session, None, None]:
    for db in get_db():
        yield db


def main() -> None:
    global SAFE_MODE_ENABLED

    parser = argparse.ArgumentParser(description="Run Journal Mood Tracker API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--demo", action="store_true", help="Seed 30 days of synthetic entries")
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Run startup safety checks (Ollama running/model pulled/DB writable)",
    )
    args = parser.parse_args()

    if args.safe_mode:
        SAFE_MODE_ENABLED = True

    init_db()

    if args.demo:
        db = next(_session_generator())
        try:
            created = seed_demo_entries(db)
            print(f"Seeded {created} demo entries.")
        finally:
            db.close()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
