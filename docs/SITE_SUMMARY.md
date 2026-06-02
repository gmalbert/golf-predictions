> **AI Onboarding Guide** — See also `copilot-instructions.md` in the repo root for coding conventions.

# Golf Predictions — Site Summary

## What This App Does

Streamlit analytics platform for PGA Tour betting predictions. Uses XGBoost and LightGBM models trained on strokes gained stats and player history to predict win probability, top-5 and top-20 finishes, and identify value bets against DraftKings odds. Includes a live tournament tracker and DK player prop calculator.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the app
streamlit run predictions.py
```

Nightly GitHub Actions pipeline refreshes player stats and tournament data. For manual refresh, run `python scrapers/espn_golf.py` and `python features/build_features.py`.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page) |
| ML | XGBoost + LightGBM ensemble, Optuna hyperparameter tuning |
| Data sources | ESPN Golf API, OWGR (world rankings, scraped), PGA Tour SG stats |
| Odds | The Odds API |
| Data storage | Parquet |
| Visualization | Plotly, BeautifulSoup |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Streamlit entry point |
| `scrapers/espn_golf.py` | ESPN Golf API: PGA schedules, leaderboards, player stats |
| `features/build_features.py` | Feature engineering: player history, OWGR merge, SG stats |
| `features/build_owgr_features.py` | World golf rankings feature builder |
| `models/train_improved_model.py` | XGBoost + LightGBM training, Optuna tuning |
| `evaluation/backtester.py` | Historical backtesting and model calibration |
| `live/tournament_tracker.py` | Live round-by-round leaderboard updates |

## Data Flow

1. **Schedule/Results**: `scrapers/espn_golf.py` → ESPN Golf API → player stats, upcoming tournaments
2. **World rankings**: `features/build_owgr_features.py` → OWGR (scraped) → ranking features
3. **Feature matrix**: `features/build_features.py` → player history, strokes gained, course history, OWGR merge → Parquet
4. **Training**: `models/train_improved_model.py` → XGBoost + LightGBM → win% / top-5 / top-20 models
5. **Value bets**: model probability vs DraftKings implied probability → edge per player
6. **UI**: Streamlit reads Parquet → renders predictions, top picks, live leaderboard

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `ODDS_API_KEY` | The Odds API — DraftKings golf odds | Required for value bets |

## Critical Conventions

- Strokes gained categories (SG: Off-Tee, Approach, Around-Green, Putting) are the most predictive features — always include them
- Course history features (course_avg_finish_l5) are high-value but require sufficient historical data per player-course pair
- Use `pathlib.Path` for all file paths

## Common Gotchas

- OWGR scraper may break when the OWGR website updates its layout — check `features/build_owgr_features.py` first when rankings seem stale
- ESPN Golf API is unofficial and may change; `scrapers/espn_golf.py` wraps these calls
- Player name normalization between ESPN, OWGR, and PGA Tour is a common source of missing matches — verify after adding new data sources
- The live tournament tracker (`live/tournament_tracker.py`) is available but may not be fully wired into the main UI
