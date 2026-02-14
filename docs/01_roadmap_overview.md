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

*Last updated: 2026-02-13*
