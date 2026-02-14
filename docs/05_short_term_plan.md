# 05 – Short-Term Plan (0–3 Months)

> **Objective:** Build a working MVP that scrapes data, trains a baseline
> model, and displays predictions in the Streamlit app.

---

## Month 1: Data Foundation

### Week 1-2: Core Scrapers

- [ ] Set up project structure:
  ```
  golf-predictions/
  ├── predictions.py          # Streamlit app
  ├── data_files/
  │   └── logo.png
  ├── scrapers/
  │   ├── __init__.py
  │   ├── shared_utils.py
  │   ├── espn_golf.py
  │   ├── pga_tour_results.py
  │   ├── wiki_winners.py
  │   └── weather.py
  ├── features/
  │   ├── __init__.py
  │   └── build_features.py
  ├── models/
  │   ├── __init__.py
  │   └── baseline_logreg.py
  ├── docs/
  └── requirements.txt
  ```

- [ ] Implement ESPN scraper (`scrapers/espn_golf.py`)
  - Scrape 2016–2026 seasons
  - Output: `data_files/espn_pga_{year}.parquet`
  
- [ ] Implement Wikipedia scraper (`scrapers/wiki_winners.py`)
  - Scrape 2010–2026 for tournament winners
  - Output: `data_files/wiki_pga_all.parquet`

- [ ] Create `requirements.txt`:
  ```
  streamlit>=1.30
  pandas>=2.0
  numpy>=1.24
  requests>=2.31
  beautifulsoup4>=4.12
  lxml>=4.9
  scikit-learn>=1.3
  xgboost>=2.0
  lightgbm>=4.0
  pyarrow>=14.0
  ```

### Week 3-4: Data Cleaning & Merging

- [ ] Standardize player names across sources
  ```python
  # Quick name normalization
  import re
  
  def normalize_name(name: str) -> str:
      """Standardize player names for cross-source matching."""
      name = name.strip()
      name = re.sub(r"\s+", " ", name)  # collapse whitespace
      name = re.sub(r"\s*(Jr\.|Sr\.|III|IV|II)\s*$", "", name)  # suffixes
      return name.lower()
  
  # Usage
  assert normalize_name("Scottie Scheffler") == "scottie scheffler"
  assert normalize_name("Davis Love III") == "davis love"
  ```

- [ ] Merge ESPN results with Wikipedia winners for validation
- [ ] Build master player table with unique IDs
- [ ] Handle edge cases: withdrawals, DQs, amateur status

---

## Month 2: Feature Engineering & Baseline Model

### Week 5-6: Feature Pipeline

- [ ] Implement `features/build_features.py` (see `04_models_and_features.md`)
- [ ] Add rolling performance features (5 and 10-event windows)
- [ ] Add course history features
- [ ] Add momentum / form indicators
- [ ] Output: `data_files/feature_matrix.parquet`

```python
# Quick validation script
import pandas as pd

df = pd.read_parquet("data_files/feature_matrix.parquet")
print(f"Shape: {df.shape}")
print(f"Years: {df['year'].min()} – {df['year'].max()}")
print(f"Players: {df['name'].nunique()}")
print(f"Tournaments: {df['tournament'].nunique()}")
print(f"\nMissing values:\n{df[FEATURES].isnull().mean().sort_values(ascending=False)}")
```

### Week 7-8: Baseline Model

- [ ] Train logistic regression (Top-10 prediction)
- [ ] Evaluate with TimeSeriesSplit (5-fold)
- [ ] Target: AUC > 0.60 (above random)
- [ ] Save model with `joblib`

```python
# Save / load model
import joblib

# After training
joblib.dump(model, "models/baseline_logreg.pkl")

# In predictions.py
model = joblib.load("models/baseline_logreg.pkl")
```

---

## Month 3: Streamlit Integration & XGBoost

### Week 9-10: Wire Up Predictions Page

- [ ] Load trained model in `predictions.py`
- [ ] Display Top-N predictions for upcoming tournament
- [ ] Show model confidence scores
- [ ] Add player stat cards

```python
# predictions.py addition – display predictions
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("models/baseline_logreg.pkl")

@st.cache_data(ttl=3600)
def get_predictions(tournament: str, top_n: int) -> pd.DataFrame:
    model = load_model()
    features = pd.read_parquet("data_files/feature_matrix.parquet")
    
    # Filter to latest feature snapshot for each player
    latest = features.groupby("name").last().reset_index()
    
    X = latest[FEATURES].fillna(0)
    latest["win_prob"] = model.predict_proba(X)[:, 1]
    
    return (
        latest[["name", "win_prob"] + FEATURES]
        .sort_values("win_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
```

### Week 11-12: Graduate to XGBoost

- [ ] Train XGBoost model (see `04_models_and_features.md`)
- [ ] Compare AUC vs baseline
- [ ] Hyperparameter tuning with `optuna`
- [ ] Feature importance analysis
- [ ] A/B display: show both model predictions side by side

---

## Deliverables by End of Month 3

| Deliverable | Status |
|-------------|--------|
| 5+ years of scraped tournament data | ☐ |
| Clean feature matrix (Parquet) | ☐ |
| Baseline logistic regression model | ☐ |
| XGBoost model with feature importance | ☐ |
| Working Streamlit app with predictions | ☐ |
| Automated scraping scripts | ☐ |

---

*Next: [06_medium_term_plan.md](06_medium_term_plan.md) – Months 3–9*
