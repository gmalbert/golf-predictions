# Fairway Oracle — GitHub Copilot Instructions

## Project Overview

**App name:** Fairway Oracle
**Purpose:** Streamlit app predicting PGA/LIV tournament winners and surfacing betting market value.
**Entry point:** `streamlit run predictions.py`
**Part of:** Betting Oracle suite

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (single-page, tabbed) |
| ML | scikit-learn, XGBoost (tournament win probability) |
| Data | pandas, OWGR PDF scraping, ESPN API |
| Odds | The Odds API (`golf_masters_tournament_winner`, etc.) |
| Config | python-dotenv (`.env` file) |
| Python | 3.9+ |

---

## File Conventions

### Key files
- `predictions.py` — entry point; sets `st.set_page_config`. Contains `TOURNAMENT_TO_EVENT_LABEL` mapping.
- `shared_utils.py` — shared helper functions.
- `owgr_scraper.py` — OWGR world ranking PDF scraper.
- `update_owgr_weekly.py` — weekly OWGR data refresh script.
- `pregenerate_predictions.py` — pre-generate predictions for all upcoming tournaments.
- `footer.py` — `add_betting_oracle_footer()` must be called at page bottom.
- `scripts/export_best_bets.py` — exports `data_files/best_bets_today.json` for Sports Picks Grid.

### Data files
- `data_files/odds_consensus_latest.csv` — latest consensus odds with columns: `event_label`, `tournament`, `player`, `best_odds`, `best_decimal`, `best_implied_raw`, `best_book`, `dk_odds`, `dk_implied_raw`, `avg_novig_prob`
- `data_files/odds_raw_latest.csv` — raw per-bookmaker odds
- `data_files/predictions_cache/` — per-player model predictions by event ID (Parquet files)
- `data_files/best_bets_today.json` — unified schema for Sports Picks Grid aggregator
- `data_files/logo.png` — app logo

### Subdirectories
- `utils/` — tournament display helpers, feature engineering
- `scrapers/` — data fetch scripts
- `scripts/` — automation scripts (export, OWGR update)

---

## Golf Domain Knowledge

### Tournament bet types
- `tournament_winner` — outright tournament winner (most common)
- Picks are player names; `game` is tournament name (e.g. `"Masters Tournament Winner"`)
- `confidence` = avg_novig_prob (no-vig model probability)
- `edge` = avg_novig_prob - dk_implied_raw (positive edge = value bet)

### Odds API sport keys
- Masters: `golf_masters_tournament_winner`
- PGA Championship: `golf_pga_championship_winner`
- US Open: `golf_us_open_winner`
- The Open: `golf_the_open_championship_winner`

### OWGR Data
- `update_owgr_weekly.py` scrapes the official OWGR PDF for world rankings
- Rankings are used as a model feature for win probability
- Run weekly before tournaments begin

### Edge calculation
- Edge = avg_novig_prob - market_implied_prob
- Positive edge = model believes player is under-priced by market
- EV_THRESHOLD = 0.03 (3% minimum edge to flag as value bet)

---

## Coding Conventions

### Streamlit patterns
```python
@st.cache_data(ttl=3600)
def load_something(path: str) -> pd.DataFrame: ...
```
- `st.set_page_config()` called ONCE in `predictions.py` only
- Use `width='stretch'` for dataframes/charts, not deprecated `use_container_width`

### Security
- API keys via `python-dotenv`: `from dotenv import load_dotenv; load_dotenv()`
- Never hardcode keys; `.env` is gitignored

### Error handling
- Check `path.exists()` before loading files
- Return empty DataFrame on API failure, never raise in data loaders
- Wrap OWGR scraping in try/except (PDF format can change)

---

## Export Script

`scripts/export_best_bets.py` reads `data_files/odds_consensus_latest.csv` and writes `data_files/best_bets_today.json` in the unified Sports Picks Grid schema:
```json
{
  "meta": {"sport": "Golf", "generated_at": "ISO8601", "model_version": "1.0.0", "season": "YYYY"},
  "bets": [{"game_date": "YYYY-MM-DD", "game": "tournament", "bet_type": "tournament_winner", ...}]
}
```
Run: `python scripts/export_best_bets.py`
