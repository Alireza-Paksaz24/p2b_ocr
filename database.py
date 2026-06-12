"""
database.py — SQLAlchemy setup with optional activation via ENABLE_DB env var.
Supports SQLite by default; swap DATABASE_URL for any SQLAlchemy-compatible backend.
"""

import os
from datetime import datetime
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

ENABLE_DB: bool = os.getenv("ENABLE_DB", "false").lower() == "true"
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./db/ocr_history.db")

engine = None
SessionLocal = None


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class OCRJob(Base):
    __tablename__ = "ocr_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Input
    original_filename = Column(String(512))
    input_type = Column(String(32))       # image | images | zip | pdf
    page_count = Column(Integer, default=0)

    # Model
    model_id = Column(String(128))
    model_name = Column(String(256))

    # Output formats requested
    output_formats = Column(JSON)          # e.g. ["md", "html", "docx"]

    # Status
    status = Column(String(32), default="pending")  # pending | processing | done | error
    error_message = Column(Text, nullable=True)
    progress = Column(Float, default=0.0)   # 0.0 – 1.0

    # Results
    output_files = Column(JSON, nullable=True)   # {format: relative_path}
    extracted_text = Column(Text, nullable=True)
    content_summary = Column(JSON, nullable=True)  # {tables, diagrams, images, text_blocks}

    # Timing
    duration_seconds = Column(Float, nullable=True)


class ModelDownload(Base):
    __tablename__ = "model_downloads"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(128), unique=True, index=True)
    model_name = Column(String(256))
    hf_tag = Column(String(512))
    downloaded_at = Column(DateTime, nullable=True)
    is_downloaded = Column(Boolean, default=False)
    local_path = Column(String(1024), nullable=True)
    size_gb = Column(Float, nullable=True)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if ENABLE_DB is true. Safe to call multiple times."""
    global engine, SessionLocal

    if not ENABLE_DB:
        return

    # Ensure SQLite directory exists
    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite:///", "").replace("sqlite://", "")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Optional[Session], None, None]:
    """FastAPI dependency — yields a DB session or None if DB is disabled."""
    if not ENABLE_DB or SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
