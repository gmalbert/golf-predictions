# Fairway Oracle — Architecture

## Overview
Streamlit app predicting PGA/LIV tournament winners and surfacing betting market value using OWGR world rankings, historical performance data, and live odds.

## Data Flow
```
OWGR PDF (weekly rankings)    The Odds API     ESPN API
        ↓                           ↓               ↓
owgr_scraper.py            fetch_today_odds.py   (fixture data)
update_owgr_weekly.py              ↓
        ↓               data_files/odds_raw_latest.csv
data_files/owgr_latest.csv data_files/odds_consensus_latest.csv
        ↓                           ↓
        ↓              Feature Engineering
        ↓                   (utils/)
        ↓               XGBoost / scikit-learn
        ↓              win probability model
        ↓                           ↓
        ↓           data_files/predictions_cache/*.parquet
        ↓                           ↓
        └─────────────────→ predictions.py (Streamlit entry)
                                    ↓
                        scripts/export_best_bets.py
                                    ↓
                        data_files/best_bets_today.json
```

## ML Model
- **scikit-learn + XGBoost** tournament winner probability
- Features: OWGR world ranking, surface history, course form, recent tournament results, strokes gained
- `avg_novig_prob` = no-vig consensus market probability (from `odds_consensus_latest.csv`)
- Edge = `avg_novig_prob − dk_implied_raw`
- EV_THRESHOLD = 0.03 (3% minimum edge)

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| OWGR | World rankings PDF | None (scraped weekly) |
| The Odds API | Tournament winner markets | `ODDS_API_KEY` |
| ESPN API | Player / event data | None (public) |

## Odds Schema (`odds_consensus_latest.csv`)
Key columns: `event_label`, `tournament`, `player`, `best_odds`, `best_decimal`, `best_implied_raw`, `best_book`, `dk_odds`, `dk_implied_raw`, `avg_novig_prob`

## Key Components
- `predictions.py` — entry, `st.set_page_config`
- `owgr_scraper.py` — OWGR PDF scraper (brittle, wrap in try/except)
- `update_owgr_weekly.py` — weekly OWGR data refresh
- `pregenerate_predictions.py` — pre-generates predictions for all upcoming tournaments
- `shared_utils.py` — shared helper functions
- `utils/` — tournament display helpers, feature engineering
- `scrapers/` — data fetch scripts
- `scripts/export_best_bets.py` — reads consensus CSV, writes `best_bets_today.json`

## Odds API Sport Keys
| Tournament | Sport Key |
|------------|-----------|
| Masters | `golf_masters_tournament_winner` |
| PGA Championship | `golf_pga_championship_winner` |
| US Open | `golf_us_open_winner` |
| The Open | `golf_the_open_championship_winner` |

## Storage
- `data_files/odds_consensus_latest.csv` — consensus odds
- `data_files/predictions_cache/` — per-event Parquet predictions
- `data_files/best_bets_today.json` — Sports Picks Grid feed
