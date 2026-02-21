# 07 – Long-Term Plan (9–18+ Months)

> **Objective:** Evolve Fairway Oracle into a production-grade prediction
> platform with deep learning, live data, automated betting analysis, and
> potential monetisation paths.

---

## Phase 1: Deep Learning & Sequence Models (Month 9-12)

### Player Career Sequence Modelling

Treat each player's career as a time series and use LSTM/Transformer
architectures to capture long-term form trends.

- [x] Build sequence dataset (player × tournament × features)
- [x] Implement LSTM model (see `04_models_and_features.md`)
- [x] Implement Transformer-based model
- [ ] Compare against gradient boosting ensemble

```python
# models/transformer_golfer.py
"""
Transformer encoder for player performance sequences.
Each player's recent N tournaments become a sequence of feature vectors.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class GolferTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        # Use [CLS]-like approach: take mean of sequence
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)
```

### Course Embedding Model

- [x] Learn course embeddings (similar to word2vec for courses)
- [x] Cluster courses by playing characteristics
- [x] Use embeddings as features for player-course fit

```python
# models/course_embeddings.py
"""
Learn dense course embeddings from player performance patterns.
Courses where similar players perform well get similar embeddings.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class CourseEmbeddingModel(nn.Module):
    """
    Matrix factorization: Player × Course → Expected finish.
    Similar to collaborative filtering in recommendation systems.
    """
    def __init__(self, n_players: int, n_courses: int, embed_dim: int = 16):
        super().__init__()
        self.player_embed = nn.Embedding(n_players, embed_dim)
        self.course_embed = nn.Embedding(n_courses, embed_dim)
        self.player_bias = nn.Embedding(n_players, 1)
        self.course_bias = nn.Embedding(n_courses, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, player_ids, course_ids):
        p = self.player_embed(player_ids)
        c = self.course_embed(course_ids)
        dot = (p * c).sum(dim=1, keepdim=True)
        return (
            dot
            + self.player_bias(player_ids)
            + self.course_bias(course_ids)
            + self.global_bias
        ).squeeze()
```

---

## Phase 2: Live Data & Automation (Month 10-14)

### Real-Time Data Pipelines

- [x] Automated weekly scraping via GitHub Actions
- [x] Live leaderboard tracking during tournaments
- [x] Auto-update predictions after each round

```yaml
# .github/workflows/weekly_scrape.yml
name: Weekly Data Scrape

on:
  schedule:
    - cron: '0 6 * * 1'  # Every Monday at 6 AM UTC
  workflow_dispatch:

jobs:
  scrape:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scrapers/espn_golf.py --current-season
      - run: python scrapers/owgr_rankings.py
      - run: python features/build_features.py
      - run: python models/predict_next_tournament.py
      - uses: actions/upload-artifact@v4
        with:
          name: weekly-predictions
          path: data_files/latest_predictions.parquet
```

### In-Tournament Live Updates

```python
# live/tournament_tracker.py
"""
Track a tournament in progress and update predictions after each round.
"""

import time
import pandas as pd
from scrapers.espn_golf import get_espn_leaderboard
from models.ensemble import GolfEnsemble


def track_tournament(event_id: str, interval_minutes: int = 30):
    """Poll ESPN for live leaderboard and re-rank predictions."""
    model = GolfEnsemble()
    model_loaded = model.load("models/ensemble.pkl")

    while True:
        print(f"\n{'='*60}")
        print(f"Updating at {pd.Timestamp.now()}")

        leaderboard = get_espn_leaderboard(event_id)
        if leaderboard.empty:
            print("No data yet. Waiting...")
        else:
            # Re-calculate features with live round data
            # ... feature pipeline update ...
            # predictions = model.predict(features)
            print(leaderboard[["name", "position", "total_score"]].head(20))

        time.sleep(interval_minutes * 60)
```

---

## Phase 3: Advanced Analytics (Month 12-15)

### Bankroll Management System

```python
# betting/bankroll.py
"""
Full Kelly criterion bankroll manager with fractional Kelly support.
"""

import pandas as pd
import numpy as np


class BankrollManager:
    def __init__(self, initial_bankroll: float = 1000.0, kelly_fraction: float = 0.25):
        self.bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction  # Use 1/4 Kelly for safety
        self.history = []

    def calculate_bet_size(
        self, model_prob: float, decimal_odds: float
    ) -> float:
        """Calculate optimal bet size using fractional Kelly."""
        b = decimal_odds - 1
        q = 1 - model_prob
        kelly = (b * model_prob - q) / b

        if kelly <= 0:
            return 0.0  # No edge – don't bet

        # Apply fraction and cap at 5% of bankroll
        bet = min(kelly * self.kelly_fraction * self.bankroll, self.bankroll * 0.05)
        return round(bet, 2)

    def place_bet(self, player: str, amount: float, odds: float, result: bool):
        """Record a bet result."""
        payout = amount * odds if result else 0
        profit = payout - amount
        self.bankroll += profit

        self.history.append({
            "player": player,
            "amount": amount,
            "odds": odds,
            "won": result,
            "profit": profit,
            "bankroll_after": self.bankroll,
        })

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.history)
        print(f"Total bets: {len(df)}")
        print(f"Win rate: {df['won'].mean():.1%}")
        print(f"Total profit: ${df['profit'].sum():.2f}")
        print(f"ROI: {df['profit'].sum() / df['amount'].sum() * 100:.1f}%")
        print(f"Current bankroll: ${self.bankroll:.2f}")
        return df
```

### Backtesting Engine

```python
# evaluation/backtester.py
"""
Walk-forward backtesting framework.
Simulates betting on every tournament from 2020–present.
"""

import pandas as pd
import numpy as np
from betting.bankroll import BankrollManager


def backtest(
    feature_matrix: pd.DataFrame,
    model,
    features: list[str],
    start_year: int = 2022,
) -> pd.DataFrame:
    """
    Walk-forward backtest: train on past, predict next tournament, repeat.
    """
    df = feature_matrix.sort_values("date").copy()
    tournaments = df[df["year"] >= start_year].groupby(["tournament", "year"])

    manager = BankrollManager(initial_bankroll=1000)
    results = []

    for (tourney, year), group in tournaments:
        # Training data: everything before this tournament
        train = df[df["date"] < group["date"].min()]
        if len(train) < 500:
            continue

        # Train model on historical data
        X_train = train[features].fillna(0)
        y_train = train["is_win"]
        model.fit(X_train, y_train)

        # Predict for this tournament
        X_test = group[features].fillna(0)
        probs = model.predict_proba(X_test)[:, 1]
        group = group.copy()
        group["pred_prob"] = probs

        # Find best bet (highest predicted probability)
        best = group.loc[group["pred_prob"].idxmax()]

        # Simulate bet (assume +2000 avg odds for winner = 21.0 decimal)
        bet_size = manager.calculate_bet_size(best["pred_prob"], 21.0)
        if bet_size > 0:
            won = best["is_win"] == 1
            manager.place_bet(best["name"], bet_size, 21.0, won)

        results.append({
            "tournament": tourney,
            "year": year,
            "predicted_winner": best["name"],
            "actual_winner": group.loc[group["is_win"] == 1, "name"].values,
            "bet_size": bet_size,
            "bankroll": manager.bankroll,
        })

    summary = manager.summary()
    return pd.DataFrame(results), summary
```

---

## Phase 4: Production & Scale (Month 15-18+)

### Deployment

- [x] Deploy Streamlit app to Streamlit Cloud (free)
- [x] Or deploy to a VPS with Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "predictions.py", "--server.port=8501"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data_files:/app/data_files
      - ./models:/app/models
    restart: unless-stopped
```

### Database Migration (SQLite → PostgreSQL)

- [x] Migrate from Parquet files to a proper database
- [x] Store predictions, bets, and results for tracking

```python
# db/schema.py
"""
Database schema for production use.
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Player(Base):
    __tablename__ = "players"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    country = Column(String)
    owgr = Column(Integer)


class TournamentResult(Base):
    __tablename__ = "tournament_results"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer)
    tournament = Column(String)
    year = Column(Integer)
    finish_position = Column(Integer)
    total_score = Column(Integer)
    sg_total = Column(Float)
    sg_putting = Column(Float)
    sg_approach = Column(Float)
    date = Column(DateTime)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    tournament = Column(String)
    year = Column(Integer)
    player_id = Column(Integer)
    model_name = Column(String)
    win_prob = Column(Float)
    top10_prob = Column(Float)
    created_at = Column(DateTime)


class Bet(Base):
    __tablename__ = "bets"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer)
    player_id = Column(Integer)
    amount = Column(Float)
    odds = Column(Float)
    result = Column(Boolean)
    profit = Column(Float)
    placed_at = Column(DateTime)


def init_db(db_url="sqlite:///data_files/fairway_oracle.db"):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()
```

### Monitoring & Alerts

```python
# monitoring/alerts.py
"""
Send alerts when value bets are found or predictions are ready.
"""

import smtplib
from email.mime.text import MIMEText


def send_email_alert(subject: str, body: str, to_email: str):
    """Send prediction alerts via email (use Gmail app password)."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "fairwayoracle@gmail.com"
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login("fairwayoracle@gmail.com", "YOUR_APP_PASSWORD")
        server.send_message(msg)


# Discord webhook alternative (simpler)
import requests

def send_discord_alert(webhook_url: str, message: str):
    requests.post(webhook_url, json={"content": message}, timeout=10)
```

---

## Potential Monetisation (Optional)

| Path | Description | Complexity |
|------|-------------|-----------|
| Free tier + Premium | Basic picks free, detailed analysis paid | Medium |
| Tipster subscription | Weekly email with top value bets | Low |
| API access | Sell model predictions via REST API | Medium |
| Affiliate links | Partner with sportsbooks | Low |
| Content / SEO | Blog posts driving traffic to site | Low |

---

## Deliverables Status

| Deliverable | File | Status |
|-------------|------|--------|
| LSTM sequence model | `models/deep_model.py` | ✅ |
| Transformer model | `models/transformer_golfer.py` | ✅ |
| Course embeddings | `models/course_embeddings.py` | ✅ |
| Weekly GitHub Actions pipeline | `.github/workflows/weekly_scrape.yml` | ✅ |
| Live tournament tracker | `live/tournament_tracker.py` | ✅ |
| Bankroll manager (Kelly) | `betting/bankroll.py` | ✅ |
| Walk-forward backtesting engine | `evaluation/backtester.py` | ✅ |
| Docker deployment | `Dockerfile`, `docker-compose.yml` | ✅ |
| Database schema (SQLite/PostgreSQL) | `db/schema.py` | ✅ |
| Email & Discord alerts | `monitoring/alerts.py` | ✅ |
| Live leaderboard auto-updates | `live/tournament_tracker.py` | ✅ |
| Full model comparison script | – | ☐ |

---

## Long-Term Feature Wishlist

- [ ] LIV Golf predictions
- [ ] European Tour (DP World Tour) predictions
- [ ] Match play / Ryder Cup models
- [ ] Prop bet predictions (top nationality, first-round leader)
- [ ] Mobile-responsive design
- [ ] User accounts with watchlists
- [ ] Social features (pick challenges, leaderboards)
- [ ] Integration with DraftKings / FanDuel DFS

---

## Final Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Fairway Oracle v2.0                      │
├──────────┬───────────────┬──────────────┬──────────────────┤
│  Scrapers│  Feature Eng. │   ML Models  │   Streamlit UI   │
│          │               │              │                  │
│ ESPN API │ Rolling stats │ LogReg       │ Predictions page │
│ PGA Tour │ Course history│ XGBoost      │ Value bets page  │
│ Wikipedia│ Weather merge │ LightGBM     │ Player cards     │
│ OWGR     │ Odds implied  │ LSTM         │ Backtest dash    │
│ Open-Met.│ Field strength│ Transformer  │ Live tracker     │
│ Odds API │ Momentum      │ Ensemble     │ Bankroll mgmt    │
├──────────┴───────────────┴──────────────┴──────────────────┤
│                     PostgreSQL / SQLite                     │
├────────────────────────────────────────────────────────────┤
│              GitHub Actions / Cron Automation               │
├────────────────────────────────────────────────────────────┤
│            Docker / Streamlit Cloud Deployment              │
└────────────────────────────────────────────────────────────┘
```

---

*This completes the Fairway Oracle roadmap. Start with
[05_short_term_plan.md](05_short_term_plan.md) and work forward!*
