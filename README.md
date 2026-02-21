# Fairway Oracle

⛳ **PGA Tournament Winner Predictions for Smarter Betting**

Fairway Oracle is a data-driven platform that predicts PGA Tour tournament winners using machine learning models trained on 5–10 years of historical data. Built for research and entertainment purposes, it helps identify value bets by comparing model predictions against market odds.

---

## Features

- **Tournament Predictions**: Top‑N winner probabilities for upcoming PGA events (XGBoost ensemble with OWGR features)
- **Player Analytics**: Strokes Gained, form trends, course history, world ranking data
- **Value Bet Detection**: Compare model probabilities vs. market odds and flag edges
- **Historical Backtesting & Bankroll Simulator**: Walk‑forward evaluation with AUC/top‑N metrics and fractional‑Kelly bankroll results
- **Live Tracking**: Optionally re‑rank predictions as tournament leaderboards update
- **Deep‑learning Sequence Models**: LSTM/Transformer training scripts for player time‑series
- **Course Embeddings**: Learn and cluster course vectors; use as future feature inputs
- **Alerts & Monitoring**: Email/Discord notifications for new predictions or value bets
- **Free Data Sources**: No paid APIs required – scrapes public PGA Tour, ESPN, OWGR, and weather data

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Front-end | Streamlit |
| Data Storage | Parquet / SQLite / PostgreSQL |
| ML Models | scikit-learn, XGBoost, LightGBM, PyTorch |
| Scraping | requests, BeautifulSoup, Selenium |
| Scheduling | GitHub Actions / cron |
| Deployment | Docker / Streamlit Cloud |

---

### Data Setup

The app starts with placeholder data. To populate real predictions:

1. **Scrape historical data:**
   ```bash
   python scrapers/espn_golf.py --start 2018 --end 2025
   ```

2. **Build player ID mapping:**
   ```bash
   python features/apply_player_ids.py         # assign and persist stable player IDs
   python features/apply_player_ids.py --validate
   ```

3. **Build features:**
   ```bash
   python features/build_features.py          # per-player tournament history
   python features/build_owgr_features.py     # add OWGR ranking columns
   python features/build_extended_features.py # tournament/weather/SG stats
   ```

4. **Train or experiment with models:**
   ```bash
   python models/train_improved_model.py       # XGBoost ensemble (default)
   python models/transformer_golfer.py         # train LSTM/Transformer sequence model
   python models/course_embeddings.py --cluster  # learn & cluster course vectors
   ```

5. **Evaluate & backtest:**
   ```bash
   python evaluation/backtester.py --start-year 2022 --kelly 0.25
   ```

6. **Run the app / live tracking:**
   ```bash
   streamlit run predictions.py
   python live/tournament_tracker.py --event-id 401703511
   ```

7. **Deploy (optional):**
   ```bash
   docker-compose up --build       # containerised Streamlit + scraper services
   ```

See the [roadmap docs](docs/) for detailed implementation steps.

---

## Project Structure

```
golf-predictions/
├── predictions.py          # Main Streamlit app
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── data_files/            # Scraped data (Parquet), player registry, and logo
├── scrapers/              # Web scraping scripts (ESPN, PGA Tour)
│   ├── shared_utils.py    # Caching, user agent rotation
│   ├── espn_golf.py       # ESPN tournament scraper
│   └── README.md          # Scraper documentation
├── features/              # Feature engineering and data prep
│   ├── player_ids.py      # Player ID mapping system
│   ├── apply_player_ids.py # Batch ID application
│   ├── build_owgr_features.py # OWGR ranking integration
│   └── README.md          # Feature engineering docs
├── models/                # ML model training (coming soon)
├── docs/                  # Roadmap and implementation guides
│   ├── 01_roadmap_overview.md
│   ├── 02_data_sources.md
│   └── ...                # 7 total documentation files
└── README.md              # This file
```

---

## Roadmap

The project follows a phased development plan:

| Phase | Duration | Focus |
|-------|----------|-------|
| [Short-Term](docs/05_short_term_plan.md) | 0–3 months | MVP with basic scraping, features, and baseline model |
| [Medium-Term](docs/06_medium_term_plan.md) | 3–9 months | Advanced models, odds integration, value bets |
| [Long-Term](docs/07_long_term_plan.md) | 9–18+ months | Deep learning, live tracking, production deployment |

See [docs/01_roadmap_overview.md](docs/01_roadmap_overview.md) for the full vision.

---

## Data Sources

All data is sourced from free, public APIs and websites:

- **PGA Tour Stats** (pgatour.com) – Strokes Gained, driving, putting
- **ESPN Golf API** (espn.com) – Tournament results, player profiles
- **Wikipedia** – Historical tournament winners
- **OWGR** (owgr.com) – World golf rankings
- **Open-Meteo** (open-meteo.com) – Weather data (free, no API key)
- **The Odds API** (the-odds-api.com) – Betting odds (free tier available)

See [docs/02_data_sources.md](docs/02_data_sources.md) for details and code samples.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.