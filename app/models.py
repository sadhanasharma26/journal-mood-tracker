from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, LargeBinary, String, Text

from app.database import Base


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String(10), unique=True, nullable=False, index=True)  # YYYY-MM-DD
    raw_text = Column(Text, nullable=False)
    sentiment_label = Column(String(20), nullable=False)  # positive / neutral / negative
    sentiment_score = Column(Float, nullable=False)  # 0.0 to 1.0
    emotions = Column(Text, nullable=False)  # JSON string of top 3 emotions with scores
    embedding = Column(LargeBinary, nullable=False)  # serialized numpy array bytes
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
