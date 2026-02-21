# 06 – Medium-Term Plan (3–9 Months)

> **Objective:** Level up from MVP to a robust, multi-model prediction system
> with betting-aware evaluation, odds integration, and automated pipelines.

---

## Phase 1: Enhanced Data Pipeline (Month 3-4)

### Add PGA Tour Stats Scraping

- [x] Scrape all Strokes Gained categories (2015–present)
- [x] Scrape driving, approach, putting, and scrambling stats
- [x] Merge with existing ESPN/Wikipedia data

```python
# scrapers/pga_stats.py (outline)
STAT_IDS = {
    "sg_total": "02675",
    "sg_off_tee": "02567",
    "sg_approach": "02568",
    "sg_around_green": "02569",
    "sg_putting": "02564",
    "driving_distance": "101",
    "driving_accuracy": "102",
    "gir_pct": "103",
    "scrambling": "130",
    "putts_per_round": "104",
}

def scrape_stat(stat_id: str, season: int) -> pd.DataFrame:
    url = f"https://www.pgatour.com/stats/detail/{stat_id}?year={season}"
    resp = polite_get(url)
    # Parse __NEXT_DATA__ JSON or HTML tables
    # ... (see 03_web_scraping.md for patterns)
```

### Add OWGR Integration

- [x] Weekly ranking snapshots (Selenium scraper)
- [x] Historical ranking at time of each tournament
- [x] Field strength calculation for each event

### Add Weather Pipeline

- [x] Automate weather fetching for tournament venues
- [x] Map tournament dates to course coordinates
- [x] Store as `data_files/weather_tournament_{year}.parquet`

---

## Phase 2: Advanced Modelling (Month 4-6)

### LightGBM Ranker Implementation

- [x] Train LambdaRank model (see `04_models_and_features.md`)
- [x] Optimize NDCG@5 and NDCG@10
- [x] Compare against XGBoost classifier

### Model Stacking / Ensemble

```python
# models/ensemble.py
"""
Stacked ensemble: combine LogReg, XGBoost, and LightGBM predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


class GolfEnsemble:
    def __init__(self):
        self.base_models = {
            "logreg": joblib.load("models/baseline_logreg.pkl"),
            "xgboost": joblib.load("models/xgboost_top10.pkl"),
            "lgbm": joblib.load("models/lgbm_top10.pkl"),
        }
        self.meta_model = LogisticRegression()

    def get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from each base model."""
        preds = []
        for name, model in self.base_models.items():
            p = model.predict_proba(X)[:, 1]
            preds.append(p)
        return np.column_stack(preds)

    def fit_meta(self, X: pd.DataFrame, y: np.ndarray):
        """Train the meta-model on base model outputs."""
        base_preds = self.get_base_predictions(X)
        self.meta_model.fit(base_preds, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Final ensemble prediction."""
        base_preds = self.get_base_predictions(X)
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def save(self, path="models/ensemble.pkl"):
        joblib.dump(self, path)
```

### Hyperparameter Optimization

```python
# models/tune_xgboost.py
"""
Optuna-based hyperparameter search for XGBoost.
"""

import optuna
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

FEATURES = [...]  # from 04_models_and_features.md
TARGET = "is_top10"


def objective(trial, X, y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    }

    tscv = TimeSeriesSplit(n_splits=4)
    aucs = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        probs = model.predict_proba(X.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[val_idx], probs))

    return np.mean(aucs)


def tune(df: pd.DataFrame, n_trials: int = 100):
    df = df.dropna(subset=FEATURES + [TARGET])
    X, y = df[FEATURES], df[TARGET]

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"Best AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params
```

---

## Phase 3: Betting Integration (Month 5-7)

### Odds Data Pipeline

- [x] Set up The Odds API (free tier)  # implemented via rotowire_odds & odds_api
- [x] Scrape pre-tournament odds weekly
- [x] Calculate implied probabilities from odds

```python
# odds/implied_prob.py
"""
Convert betting odds to implied probabilities.
"""


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def decimal_to_implied(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / odds


def remove_vig(probs: list[float]) -> list[float]:
    """Remove bookmaker's vig (overround) to get true probabilities."""
    total = sum(probs)
    return [p / total for p in probs]


# Example
odds = {"Scheffler": -150, "McIlroy": +800, "Rahm": +1200}
implied = {name: american_to_implied(o) for name, o in odds.items()}
print(implied)
# {'Scheffler': 0.6, 'McIlroy': 0.111, 'Rahm': 0.077}
```

### Value Bet Detection

```python
# odds/value_finder.py
"""
Compare model predictions against market odds to find value bets.
"""

import pandas as pd


def find_value_bets(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_edge: float = 0.05,
) -> pd.DataFrame:
    """
    Find bets where our model gives > min_edge advantage over the market.
    
    predictions: name, model_prob
    odds: name, american_odds, implied_prob
    """
    merged = predictions.merge(odds, on="name", how="inner")
    merged["edge"] = merged["model_prob"] - merged["implied_prob"]
    merged["is_value"] = merged["edge"] > min_edge

    value = merged[merged["is_value"]].sort_values("edge", ascending=False)
    
    print(f"Found {len(value)} value bets (edge > {min_edge*100:.0f}%)")
    for _, row in value.iterrows():
        print(
            f"  {row['name']}: Model {row['model_prob']:.1%} vs "
            f"Market {row['implied_prob']:.1%} → Edge {row['edge']:.1%} "
            f"(Odds: {row['american_odds']:+d})"
        )
    return value
```

---

## Phase 4: Enhanced UI (Month 7-9)

### Streamlit Multi-Page App

- [ ] Add sidebar navigation
- [ ] Pages: Predictions, Player Cards, Model Performance, Value Bets
- [ ] Historical backtesting dashboard

```python
# pages/value_bets.py  (Streamlit multi-page support)
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Value Bets – Fairway Oracle", page_icon="💰")

st.title("💰 Value Bet Finder")
st.markdown("Bets where our model sees edge over the market.")

# Load latest predictions and odds
# predictions = pd.read_parquet("data_files/latest_predictions.parquet")
# odds = pd.read_parquet("data_files/latest_odds.parquet")
# value = find_value_bets(predictions, odds)

# st.dataframe(value[["name", "model_prob", "implied_prob", "edge", "american_odds"]])

st.info("Connect odds pipeline to activate this page.")
```

### Visualization Additions

- [ ] Win probability bar charts
- [ ] Player form trend lines
- [ ] Course history heatmaps
- [ ] Model performance over time

```python
# Example: Plotly chart for predictions
import plotly.express as px

def plot_predictions(df: pd.DataFrame, top_n: int = 20):
    top = df.head(top_n).sort_values("win_prob")
    fig = px.bar(
        top, x="win_prob", y="name",
        orientation="h",
        title=f"Top {top_n} – Win Probability",
        labels={"win_prob": "Predicted Probability", "name": "Player"},
        color="win_prob",
        color_continuous_scale="Greens",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    return fig

# In Streamlit:
# st.plotly_chart(plot_predictions(predictions_df))
```

---

## Deliverables by End of Month 9

| Deliverable | Status |
|-------------|--------|
| Strokes Gained data pipeline (2015–present) | ✅ |
| OWGR integration | ✅ |
| Weather data pipeline | ✅ |
| LightGBM ranker model | ✅ |
| Stacked ensemble model | ☐ |
| Optuna hyperparameter tuning | ☐ |
| Betting odds pipeline | ✅ |
| Value bet detection system | ✅ |
| Multi-page Streamlit app | ✅ |
| Backtesting dashboard | ☐ |

---

*Next: [07_long_term_plan.md](07_long_term_plan.md) – Months 9–18+*
