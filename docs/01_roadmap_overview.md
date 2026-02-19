# Fairway Oracle – Project Roadmap

> **Goal:** Build a data-driven PGA tournament prediction system that provides
> actionable betting insights using 5–10 years of historical data, modern ML
> models, and a polished Streamlit front-end.

---

## Document Index

| # | Document | Description |
|---|----------|-------------|
| 01 | [Roadmap Overview](01_roadmap_overview.md) | This file – high-level vision & plan |
| 02 | [Data Sources](02_data_sources.md) | Free APIs, datasets, and scraping targets |
| 03 | [Web Scraping Guide](03_web_scraping.md) | Scraping code samples & best practices |
| 04 | [Models & Features](04_models_and_features.md) | Suggested ML models and feature engineering |
| 05 | [Short-Term Plan](05_short_term_plan.md) | 0–3 months: MVP |
| 06 | [Medium-Term Plan](06_medium_term_plan.md) | 3–9 months: Enhanced models & UI |
| 07 | [Long-Term Plan](07_long_term_plan.md) | 9–18+ months: Full production system |

---

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Data Layer  │────▶│  ML Pipeline │────▶│  Streamlit App   │
│              │     │              │     │  (predictions.py)│
│ • Scrapers   │     │ • Features   │     │                  │
│ • APIs       │     │ • Training   │     │ • Predictions    │
│ • CSV/Parquet│     │ • Evaluation │     │ • Stats          │
│              │     │ • Inference  │     │ • Betting Odds   │
└──────────────┘     └──────────────┘     └──────────────────┘
```

## Key Principles

1. **Free data only** – No paid APIs or subscriptions required at any tier.
2. **Reproducible pipelines** – Every data transformation is scripted, versioned, and testable.
3. **Incremental complexity** – Start simple (logistic regression), add complexity as data grows.
4. **Betting-aware metrics** – Optimize for ROI, not just accuracy.
5. **Historical depth** – Target 5–10 years of tournament-level & round-level data.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Front-end | Streamlit |
| Data storage | Parquet / SQLite |
| ML | scikit-learn → XGBoost → LightGBM → PyTorch |
| Scraping | `requests` + `BeautifulSoup` / `Selenium` |
| Scheduling | `schedule` / GitHub Actions / cron |
| Version control | Git + GitHub |

---

## Recommended next steps (short & medium term)

> Practical action items to complete after Tier‑2 (PGA stats) and Tier‑3 (weather) integration.

1. ✅ **Immediate / High priority (0–2 weeks)**
   - ✅ Update CI & tests to prefer the extended dataset (`data_files/espn_with_extended_features.parquet`) — `models/test_model_v2.py` updated to prefer extended parquet.  
   - Add unit tests for SG merges and weather enrichment (validate joins, percent parsing, and coverage) — files: `scrapers/pga_stats.py`, `features/build_extended_features.py`.  
   - Open a short PR that documents the new datasets and model v3 artifacts (`models/saved_models/*`) (artifacts added; PR pending).

2. 🔧 **Medium priority (2–6 weeks)**
   - ✅ Implement canonical player‑ID mapping / stronger name normalization to raise SG‑join coverage — implemented (`features/player_ids.py`, `features/apply_player_ids.py`).  
   - Add remaining features from `docs/04_models_and_features.md` (e.g. `course_length_fit`, `course_history_sg`, `momentum_score`) and corresponding tests.  
   - Wire model **v3** into the Streamlit UI and prediction endpoints (`predictions.py`) and update user docs.  
   - ✅ RotoWire odds scraping + Streamlit odds UI (outrights, best-book, DK column, Value Bet) — implemented (`scrapers/rotowire_odds.py`, `predictions.py`).

3. 📅 **Longer term / backlog**
   - Schedule automated scrapes (PGA stats + weather) via GitHub Actions/cron and add monitoring/alerts.  
   - Expand course-level metadata (altitude, grass-specific effects) and run ablation experiments.  
   - Evaluate premium data integrations (DataGolf) as optional uplift sources.

### Quick wins (do today)
- ✅ Ensure `models/test_model_v2.py` prefers the extended parquet (done); CI/tests updated.  
- Add 2–3 unit tests covering SG parsing and name‑based merges.  
- ✅ Canonical player‑id mapping implemented (`features/player_ids.py`) — create issue to track follow-ups.  
- ✅ RotoWire odds + UI integrated (quick value bet indicator added).

---

*Last updated: 2026-02-18*
