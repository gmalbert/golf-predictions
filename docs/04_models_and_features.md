# 04 – Suggested Models & Feature Engineering

> A progressive modelling strategy: start simple, add complexity as data and
> understanding grow. Every model section includes a ready-to-run code sample.

---

## Feature Categories

> **Note:** a ✅ indicates the feature has been wired into the current
> feature‑building pipeline (`features/build_features.py` or
> `features/build_extended_features.py`).  Several rows below are still
> placeholders – they appear in the plan but the code hasn’t generated them
> yet.  (See 📝 status comments on individual rows.)


### A. Player Skill Features (most important)

| Feature | Source | Description |
|---------|--------|-------------|
| ✅ `sg_total` | PGA Tour | Strokes Gained: Total (rolling avg) |
| ✅ `sg_off_tee` | PGA Tour | Strokes Gained: Off the Tee |
| ✅ `sg_approach` | PGA Tour | Strokes Gained: Approach the Green |
| ✅ `sg_around_green` | PGA Tour | Strokes Gained: Around the Green |
| ✅ `sg_putting` | PGA Tour | Strokes Gained: Putting |
| ✅ `driving_distance` | PGA Tour | Average driving distance |
| ✅ `driving_accuracy` | PGA Tour | Fairways hit % (now wired into feature pipeline) |
| ✅ `gir_pct` | PGA Tour | Greens in Regulation % |
| ✅ `scrambling_pct` | PGA Tour | Up-and-down % from missed greens |
| ✅ `putts_per_gir` | PGA Tour | Putts per green in regulation |
| ✅ `birdie_avg` | PGA Tour | Birdies per round |
| ✅ `scoring_avg` | PGA Tour | Adjusted scoring average (now wired into feature pipeline) |
| ✅ `top10_pct` | Calculated | % of events finishing Top-10 (rolling) |
| ✅ `cut_pct` | Calculated | % of cuts made (rolling) |
| ✅ `win_pct` | Calculated | % of wins (rolling) |

### B. Form / Momentum Features

| Feature | Window | Description |
|---------|--------|-------------|
| ✅ `recent_sg_total_5` | Last 5 events | Rolling mean SG:Total |
| ✅ `recent_sg_total_10` | Last 10 events | Rolling mean SG:Total |
| ✅ `recent_finish_avg_5` | Last 5 events | Average finish position |
| ✅ `recent_top10_count_10` | Last 10 events | Count of Top-10 finishes |
| ✅ `momentum_score` | Last 5 vs last 20 | SG improvement trend |
| ✅ `days_since_last_event` | Calculated | Rest / fatigue indicator |
| ✅ `events_in_last_30d` | Calculated | Workload |

### C. Course Fit Features

| Feature | Source | Description |
|---------|--------|-------------|
| ✅ `course_history_avg_finish` | Historical | Past finishes at this course |
| ✅ `course_history_sg` | Historical | SG at this specific course |
| ✅ `course_length_fit` | Course data | z_drive × z_yardage interaction — player distance vs course length |
| ✅ `course_type` | Manual tag | Links / parkland / desert / tropical |
| ✅ `past_appearances` | Calculated | # of times played this course |
| ✅ `grass_fit` | Course data | Expanding per-player bermuda vs bentgrass advantage (was `bermuda_vs_bent`) |

### D. Tournament Context Features

| Feature | Source | Description |
|---------|--------|-------------|
| ✅ `field_strength` | OWGR | Average OWGR of field |
| ✅ `purse_size_m` | Course metadata | Tournament purse in $M (continuous, from purse_tier mapping) |
| ✅ `is_major` | Flag | Major championship indicator |
| ✅ `is_playoff_event` | Flag | FedEx Cup playoff event |
| ~~`tournament_round`~~ | Leaderboard | Current round (1-4) — **not applicable** for pre-tournament model |

### E. Environmental Features

| Feature | Source | Description |
|---------|--------|-------------|
| ✅ `wind_speed_avg` | Open-Meteo | Average wind during rounds |
| ✅ `wind_gust_max` | Open-Meteo | Max gusts |
| ✅ `temperature` | Open-Meteo | Average temperature |
| ✅ `precipitation_mm` | Open-Meteo | Rain accumulation |
| ✅ `altitude_ft` | Course data | Course elevation |

---

## Feature Engineering Code ✅

*Implemented in `features/build_features.py` (see the script in the repo).* 

```python
# features/build_features.py
"""
Build feature matrix from raw scraped data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data_files")


def load_results() -> pd.DataFrame:
    """Load all scraped PGA results into one DataFrame."""
    files = sorted(DATA_DIR.glob("espn_pga_*.parquet"))
    if not files:
        files = sorted(DATA_DIR.glob("pga_results_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling performance features grouped by player."""
    df = df.sort_values(["name", "date"]).copy()

    # Convert finish position to numeric
    df["finish_numeric"] = pd.to_numeric(
        df["position"].str.replace("T", "").str.strip(),
        errors="coerce",
    )

    grouped = df.groupby("name")

    # Rolling average finish
    df[f"avg_finish_{window}"] = grouped["finish_numeric"].transform(
        lambda x: x.rolling(window, min_periods=3).mean()
    )

    # Rolling top-10 count
    df["is_top10"] = (df["finish_numeric"] <= 10).astype(int)
    df[f"top10_count_{window}"] = grouped["is_top10"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

    # Rolling win count
    df["is_win"] = (df["finish_numeric"] == 1).astype(int)
    df[f"win_count_{window}"] = grouped["is_win"].transform(
        lambda x: x.rolling(window, min_periods=1).sum()
    )

    # Cut-made rate
    df["made_cut"] = df["finish_numeric"].notna().astype(int)
    df[f"cut_rate_{window}"] = grouped["made_cut"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    return df


def add_course_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical performance at each course/tournament."""
    df = df.sort_values(["name", "tournament", "date"]).copy()

    # Past appearances at this tournament
    df["past_appearances"] = df.groupby(["name", "tournament"]).cumcount()

    # Historical avg finish at this tournament
    df["course_avg_finish"] = (
        df.groupby(["name", "tournament"])["finish_numeric"]
        .transform(lambda x: x.expanding().mean().shift(1))
    )

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term vs long-term form trend."""
    df = df.sort_values(["name", "date"]).copy()
    grouped = df.groupby("name")

    short = grouped["finish_numeric"].transform(
        lambda x: x.rolling(5, min_periods=2).mean()
    )
    long = grouped["finish_numeric"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # Negative = improving (lower finish = better)
    df["momentum"] = short - long
    return df


def build_feature_matrix() -> pd.DataFrame:
    """Full pipeline: load → engineer → output."""
    df = load_results()
    df = add_rolling_features(df, window=10)
    df = add_rolling_features(df, window=5)
    df = add_course_history(df)
    df = add_momentum(df)

    out = DATA_DIR / "feature_matrix.parquet"
    df.to_parquet(out, index=False)
    print(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols → {out}")
    return df


if __name__ == "__main__":
    build_feature_matrix()
```

---

## Suggested Models (Progressive Complexity)

### Tier 1 — Baseline (Week 1-2) ✅ implemented (`models/baseline_logreg.py`)

**Logistic Regression / Ordinal Regression**

Simple, interpretable, fast to iterate. Use for "will this player finish Top-10?" (binary).

```python
# models/baseline_logreg.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss
import pandas as pd
import numpy as np

FEATURES = [
    "avg_finish_10", "top10_count_10", "cut_rate_10",
    "course_avg_finish", "past_appearances", "momentum",
]
TARGET = "is_top10"  # binary: finished top 10?


def train_baseline(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].values
    y = df[TARGET].values

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(max_iter=1000, C=0.1)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        aucs.append(auc)
        print(f"  Fold AUC: {auc:.4f}")

    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return model
```

---

### Tier 2 — Gradient Boosting (Month 1-2) ✅ implemented (`models/train_improved_model.py` + XGBoost)

**XGBoost / LightGBM** – The workhorse of tabular prediction.

```python
# models/xgboost_model.py
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

FEATURES = [
    "avg_finish_10", "avg_finish_5",
    "top10_count_10", "top10_count_5",
    "win_count_10", "cut_rate_10", "cut_rate_5",
    "course_avg_finish", "past_appearances",
    "momentum",
]
TARGET = "is_top10"


def train_xgboost(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        aucs.append(auc)
        models.append(model)
        print(f"  Fold {fold+1} AUC: {auc:.4f}")

    print(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    # Feature importance
    best = models[np.argmax(aucs)]
    importance = pd.Series(
        best.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(importance.to_string())

    return best
```

---

### Tier 3 — Ensemble & Ranking Models (Month 3-6) ✅ implemented (`models/lgbm_ranker.py`)

**LightGBM Ranker** – Directly optimize for "who finishes highest?"

```python
# models/lgbm_ranker.py
"""
LambdaRank model: learns to rank players within a tournament.
Better than classification for predicting finish order.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np

FEATURES = [
    "avg_finish_10", "avg_finish_5",
    "top10_count_10", "win_count_10",
    "cut_rate_10", "course_avg_finish",
    "past_appearances", "momentum",
]


def train_ranker(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + ["finish_numeric"]).copy()

    # Lower finish = better; invert for ranking relevance
    max_finish = df["finish_numeric"].max()
    df["relevance"] = (max_finish - df["finish_numeric"]).clip(lower=0)

    # Group sizes (players per tournament)
    groups = df.groupby(["tournament", "year"]).size().values

    X = df[FEATURES].values
    y = df["relevance"].values

    # Simple train/test split by time
    split = int(len(groups) * 0.8)
    cumsum = np.cumsum(groups)
    split_idx = cumsum[split]

    X_tr, X_val = X[:split_idx], X[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]
    g_tr, g_val = groups[:split], groups[split:]

    train_data = lgb.Dataset(X_tr, y_tr, group=g_tr, feature_name=FEATURES)
    val_data = lgb.Dataset(X_val, y_val, group=g_val, feature_name=FEATURES)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    print(f"\nBest NDCG@10: {model.best_score['valid_0']['ndcg@10']:.4f}")
    return model
```

---

### Tier 4 — Deep Learning (Month 6+) ✅ implemented (`models/deep_model.py`)

**Neural network for sequence modelling** – Treat a player's tournament history
as a time series.

```python
# models/deep_model.py (sketch – expand during long-term phase)
"""
LSTM / Transformer model that treats each player's career
as a sequence of tournament performances.
"""

import torch
import torch.nn as nn


class GolferLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])  # use last hidden state
        return out.squeeze(-1)


# Usage sketch:
# model = GolferLSTM(input_dim=len(FEATURES))
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# criterion = nn.BCELoss()
# ... training loop ...
```

---

## Model Evaluation: Betting-Aware Metrics

Standard ML metrics (AUC, accuracy) don't capture betting value. Add these:

```python
# evaluation/betting_metrics.py
"""
Betting-specific evaluation metrics.
"""

import numpy as np
import pandas as pd


def calculate_roi(predictions: pd.DataFrame) -> float:
    """
    Calculate ROI for outright winner bets.
    predictions must have columns: predicted_prob, actual_win, american_odds
    """
    df = predictions.copy()
    # Kelly criterion: bet fraction = (bp - q) / b
    # b = decimal odds - 1, p = predicted prob, q = 1 - p
    df["decimal_odds"] = df["american_odds"].apply(
        lambda x: (x / 100) + 1 if x > 0 else (100 / abs(x)) + 1
    )
    df["b"] = df["decimal_odds"] - 1
    df["kelly_fraction"] = (
        (df["b"] * df["predicted_prob"] - (1 - df["predicted_prob"])) / df["b"]
    ).clip(lower=0)

    # Only bet when Kelly > 0 (edge exists)
    bets = df[df["kelly_fraction"] > 0].copy()
    if bets.empty:
        return 0.0

    bets["wager"] = bets["kelly_fraction"] * 100  # $100 bankroll
    bets["payout"] = bets["actual_win"] * bets["wager"] * bets["decimal_odds"]
    bets["profit"] = bets["payout"] - bets["wager"]

    total_wagered = bets["wager"].sum()
    total_profit = bets["profit"].sum()
    roi = total_profit / total_wagered * 100
    print(f"Bets placed: {len(bets)}")
    print(f"Total wagered: ${total_wagered:.2f}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.1f}%")
    return roi


def value_score(predicted_prob: float, implied_prob: float) -> float:
    """
    How much edge our model sees vs the market.
    Positive = value bet.
    """
    return predicted_prob - implied_prob
```

---

## Model Selection Summary

| Phase | Model | Target | Why |
|-------|-------|--------|-----|
| Baseline | Logistic Regression | Top-10 (binary) | Fast, interpretable |
| Core | XGBoost / LightGBM Classifier | Top-10 / Top-5 / Winner | Best for tabular data |
| Advanced | LightGBM LambdaRank | Finish order | Directly optimizes ranking |
| Ensemble | Stacked (LR + XGB + LGBM) | Weighted avg | Reduces variance |
| Deep | LSTM / Transformer | Sequence of finishes | Captures career trajectories |

---

*Next: [05_short_term_plan.md](05_short_term_plan.md) – 0–3 month MVP plan*
