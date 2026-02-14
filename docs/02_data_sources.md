# 02 – Free Data Sources for PGA Predictions

> Every source below is **free** (as of Feb 2026). Sources are ranked by
> data depth, reliability, and ease of access.

---

## 1. PGA Tour Official Stats (pgatour.com)

**URL:** `https://www.pgatour.com/stats`  
**Coverage:** 2004–present (some stats back to 1980)  
**What you get:**
- Strokes Gained (all categories)
- Driving Distance / Accuracy
- Greens in Regulation
- Putting stats
- Scoring averages
- Tournament results, cuts made, finishes

**Access method:** Web scraping (see `03_web_scraping.md`)

```python
# Quick example – scrape SG: Total leaderboard
import requests
from bs4 import BeautifulSoup

url = "https://www.pgatour.com/stats/detail/02675"  # SG: Total
headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(url, headers=headers, timeout=30)
soup = BeautifulSoup(resp.text, "html.parser")
# Parse table rows – see 03_web_scraping.md for full implementation
```

---

## 2. ESPN PGA Leaderboards & Results

**URL:** `https://www.espn.com/golf/`  
**Coverage:** ~2001–present  
**What you get:**
- Tournament leaderboards (round-by-round scores)
- Player profiles
- World rankings history

```python
# ESPN leaderboard API (unofficial JSON endpoint)
import requests

tournament_id = "401580333"  # Example tournament ID
url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/leaderboard?event={tournament_id}"
resp = requests.get(url, timeout=30)
data = resp.json()

for competitor in data["events"][0]["competitions"][0]["competitors"][:10]:
    name = competitor["athlete"]["displayName"]
    score = competitor.get("score", "N/A")
    print(f"{name}: {score}")
```

---

## 3. DataGolf (datagolf.com) – Free Tier

**URL:** `https://datagolf.com`  
**Coverage:** 2015–present (limited free tier)  
**What you get (free):**
- Pre-tournament predictions (viewable, not bulk downloadable)
- Historical model outputs
- Skill decomposition data

**Note:** Their paid API is excellent but the free tier gives you enough to
benchmark your own model against theirs.

---

## 4. Golf Stats & Records – Wikipedia

**URL:** `https://en.wikipedia.org/wiki/List_of_PGA_Tour_events`  
**Coverage:** Full historical archive  
**What you get:**
- Tournament winners (decades of history)
- Course info, locations, purses
- Major championship results back to the 1800s

```python
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/2025_PGA_Tour"
resp = requests.get(url, timeout=30)
soup = BeautifulSoup(resp.text, "html.parser")

tables = soup.find_all("table", {"class": "wikitable"})
# Parse tournament schedule table
for row in tables[0].find_all("tr")[1:]:
    cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
    if cols:
        print(cols)
```

---

## 5. OWGR – Official World Golf Ranking

**URL:** `https://www.owgr.com/ranking`  
**Coverage:** 1986–present  
**What you get:**
- Weekly world rankings
- Points breakdown
- Events played count

```python
# OWGR provides downloadable ranking files
import requests
import pandas as pd

# Rankings are available as downloadable CSVs
url = "https://www.owgr.com/ranking?pageNo=1&pageSize=200&country=All"
# Note: OWGR may require Selenium for dynamic content – see scraping guide
```

---

## 6. Golf Course Databases

### a) ShotLink Data (PGA Tour)
- **Access:** Limited – PGA Tour restricts bulk access
- **Alternative:** Scrape aggregate stats from pgatour.com stats pages

### b) GolfPass / Golf Advisor (course details)
- **URL:** `https://www.golfadvisor.com`
- **What you get:** Course ratings, yardage, par, slope, reviews

---

## 7. Weather Data (free)

Weather is a significant factor in golf. Use it as a feature.

### a) Open-Meteo (no API key required)
**URL:** `https://open-meteo.com/`

```python
import requests

# Historical weather for Augusta National (Masters week)
params = {
    "latitude": 33.5021,
    "longitude": -82.0227,
    "start_date": "2024-04-11",
    "end_date": "2024-04-14",
    "hourly": "temperature_2m,windspeed_10m,precipitation",
}
resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
weather = resp.json()
print(weather["hourly"]["temperature_2m"][:8])  # first 8 hours
```

### b) Visual Crossing (free tier – 1000 req/day)
**URL:** `https://www.visualcrossing.com/weather-api`

---

## 8. Betting Odds Data

### a) The Odds API (free tier – 500 req/month)
**URL:** `https://the-odds-api.com/`

```python
import requests

API_KEY = "YOUR_FREE_KEY"  # Sign up at the-odds-api.com
url = "https://api.the-odds-api.com/v4/sports/golf_pga/odds/"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "outrights",
    "oddsFormat": "american",
}
resp = requests.get(url, params=params, timeout=30)
odds_data = resp.json()
for bookmaker in odds_data:
    print(bookmaker["bookmakers"][0]["title"])
```

### b) Scraping Odds from Public Sites
- **OddsShark:** `https://www.oddsshark.com/golf/pga`
- **VegasInsider:** `https://www.vegasinsider.com/golf/odds/futures/`

---

## 9. GitHub Open Datasets

| Repository | Data |
|-----------|------|
| `golfstats/pga-tour-data` | Historical scores & stats |
| Various Kaggle datasets | Search "PGA Tour" on Kaggle |

```python
# Load a Kaggle CSV (example)
import pandas as pd

df = pd.read_csv("data_files/pga_tour_historical.csv")
print(df.columns.tolist())
print(f"Years covered: {df['year'].min()} – {df['year'].max()}")
```

---

## 10. SportsReference / Golf-Reference

**URL:** `https://www.sports-reference.com/golf/`  
**Coverage:** Extensive historical data  
**What you get:**
- Player career stats
- Tournament results
- Year-by-year performance

---

## Data Coverage Summary

| Source | Years | Granularity | Access |
|--------|-------|-------------|--------|
| PGA Tour Stats | 2004–now | Round/hole level aggregates | Scrape |
| ESPN | 2001–now | Round-by-round scores | API/Scrape |
| Wikipedia | 1916–now | Tournament winners | Scrape |
| OWGR | 1986–now | Weekly rankings | Scrape/Download |
| Open-Meteo | 1940–now | Hourly weather | Free API |
| The Odds API | Current | Pre-tournament odds | Free API (limited) |
| Sports Reference | 1958–now | Seasonal stats | Scrape |

> **Combined target:** 5–10 years of detailed round-level data (2016–2026),  
> with tournament-winner data going back much further.

---

*Next: [03_web_scraping.md](03_web_scraping.md) – Full scraping code samples*
