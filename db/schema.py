"""
Database schema for production use (SQLite by default; swap URL for PostgreSQL).

Usage
-----
    from db.schema import init_db, Player, TournamentResult, Prediction, Bet

    session = init_db()                    # SQLite at data_files/fairway_oracle.db

    # PostgreSQL (set DATABASE_URL env var):
    #   init_db("postgresql://user:pass@host/fairway")
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DEFAULT_DB = (
    "sqlite:///" + str(Path(__file__).resolve().parent.parent / "data_files" / "fairway_oracle.db")
)


class Base(DeclarativeBase):
    pass


# ── Tables ────────────────────────────────────────────────────────────────────

class Player(Base):
    __tablename__ = "players"

    id          = Column(Integer, primary_key=True)
    player_id   = Column(String, unique=True, nullable=False)  # e.g. ESPN ID
    name        = Column(String, nullable=False)
    country     = Column(String)
    owgr        = Column(Integer)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TournamentResult(Base):
    __tablename__ = "tournament_results"

    id              = Column(Integer, primary_key=True)
    player_id       = Column(String, nullable=False)
    tournament      = Column(String, nullable=False)
    year            = Column(Integer)
    finish_position = Column(Integer)
    total_score     = Column(Integer)
    sg_total        = Column(Float)
    sg_putting      = Column(Float)
    sg_approach     = Column(Float)
    sg_off_tee      = Column(Float)
    sg_around_green = Column(Float)
    date            = Column(DateTime)
    source          = Column(String, default="espn")


class Prediction(Base):
    __tablename__ = "predictions"

    id          = Column(Integer, primary_key=True)
    tournament  = Column(String, nullable=False)
    year        = Column(Integer)
    player_id   = Column(String)
    player_name = Column(String)
    model_name  = Column(String, default="winner_predictor_v2")
    win_prob    = Column(Float)
    top10_prob  = Column(Float)
    pred_rank   = Column(Integer)   # 1 = highest predicted probability
    created_at  = Column(DateTime, default=datetime.utcnow)


class Bet(Base):
    __tablename__ = "bets"

    id            = Column(Integer, primary_key=True)
    player_id     = Column(String)
    player_name   = Column(String)
    tournament    = Column(String)
    year          = Column(Integer)
    amount        = Column(Float)
    decimal_odds  = Column(Float)
    model_prob    = Column(Float)
    edge          = Column(Float)    # model_prob − implied_prob
    result        = Column(Boolean)  # None until settled
    payout        = Column(Float)
    profit        = Column(Float)
    kelly_frac    = Column(Float)
    placed_at     = Column(DateTime, default=datetime.utcnow)
    settled_at    = Column(DateTime, nullable=True)
    notes         = Column(String, default="")


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db(db_url: str | None = None):
    """
    Create all tables if they don't exist and return a Session factory.

    Parameters
    ----------
    db_url : SQLAlchemy connection string.
             Reads DATABASE_URL env var if not provided, else uses SQLite default.
    """
    url = db_url or os.getenv("DATABASE_URL", _DEFAULT_DB)
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


if __name__ == "__main__":
    session = init_db()
    print("[OK] Database initialised →", session.bind.url)
    print("Tables:", ", ".join(Base.metadata.tables.keys()))
