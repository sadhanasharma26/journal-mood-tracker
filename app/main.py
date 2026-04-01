import argparse
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Generator

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.embeddings import (
    build_faiss_index,
    generate_embedding,
    semantic_search,
    serialize_embedding,
)
from app.insights import generate_insight
from app.models import JournalEntry
from app.sentiment import analyze_entry


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Journal Mood Tracker API",
    version="0.1.0",
    lifespan=lifespan,
)


class EntryCreate(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    text: str = Field(..., min_length=1)


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
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    entries = db.query(JournalEntry).order_by(JournalEntry.date.asc()).all()
    entry_dicts = [{"id": e.id, "embedding": e.embedding} for e in entries]

    index, id_map = build_faiss_index(entry_dicts)
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
    parser = argparse.ArgumentParser(description="Run Journal Mood Tracker API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--demo", action="store_true", help="Seed 30 days of synthetic entries")
    args = parser.parse_args()

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
