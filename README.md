# Fairway Oracle

⛳ **PGA Tournament Winner Predictions for Smarter Betting**

Fairway Oracle is a data-driven platform that predicts PGA Tour tournament winners using machine learning models trained on 5–10 years of historical data. Built for research and entertainment purposes, it helps identify value bets by comparing model predictions against market odds.

> **⚠️ Disclaimer:** This is not financial advice. Gambling involves risk. Use at your own discretion.

---

## Features

- **Tournament Predictions**: Top-N winner predictions for upcoming PGA events
- **Player Analytics**: Strokes Gained, form trends, course history
- **Value Bet Detection**: Compare model probabilities vs. market odds
- **Historical Backtesting**: Evaluate model performance over past tournaments
- **Live Tracking**: Update predictions during tournament rounds
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

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gmalbert/golf-predictions.git
   cd golf-predictions
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run predictions.py
   ```
   Open http://localhost:8501 in your browser.

### Data Setup

The app starts with placeholder data. To populate real predictions:

1. **Scrape historical data:**
   ```bash
   # Scrape ESPN tournament results (2022 season)
   python scrapers/espn_golf.py --year 2022
   
   # Scrape multiple years
   python scrapers/espn_golf.py --start 2018 --end 2024
   ```

2. **Build player ID mapping:**
   ```bash
   # Creates stable player IDs from scraped data
   python features/apply_player_ids.py
   
   # Validate player IDs
   python features/apply_player_ids.py --validate
   ```

3. **Build features:**
   ```bash
   python features/build_features.py
   ```

4. **Train a model:**
   ```bash
   python models/baseline_logreg.py
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

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

- Use `black` for code formatting
- Add tests for new features
- Update docs for significant changes

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author:** [Your Name]
- **GitHub:** [gmalbert](https://github.com/gmalbert)
- **Project:** [golf-predictions](https://github.com/gmalbert/golf-predictions)

---

*Built with ❤️ for golf analytics and responsible betting research.*
