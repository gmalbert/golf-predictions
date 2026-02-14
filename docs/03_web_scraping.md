# 03 – Web Scraping Guide

> Complete, copy-paste-ready scraping code for every major free PGA data
> source.  All code uses `requests` + `BeautifulSoup` unless a site requires
> JavaScript rendering (in which case `Selenium` examples are provided).

---

## Prerequisites

```bash
pip install requests beautifulsoup4 pandas lxml selenium webdriver-manager
```

```python
# shared_utils.py – helpers used by every scraper
import time
import random
import requests
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
DATA_DIR = Path("data_files")
DATA_DIR.mkdir(exist_ok=True)


def polite_get(url: str, **kwargs) -> requests.Response:
    """GET with random delay to avoid hammering servers."""
    time.sleep(random.uniform(1.5, 3.5))
    resp = requests.get(url, headers=HEADERS, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp
```

---

## 1. PGA Tour – Tournament Results Scraper

Scrapes leaderboard results for a given season from pgatour.com.

```python
# scrapers/pga_tour_results.py
"""
Scrape PGA Tour season results (tournament winners + leaderboards).
PGA Tour uses a GraphQL / Next.js API under the hood.
"""

import json
import pandas as pd
from shared_utils import polite_get, DATA_DIR


def get_schedule(year: int) -> list[dict]:
    """Fetch the PGA Tour schedule for a given year."""
    url = (
        f"https://www.pgatour.com/tournaments/"
        f"schedule.html"
    )
    # PGA Tour exposes schedule data via embedded JSON
    resp = polite_get(url)
    # Look for the __NEXT_DATA__ script tag
    import re
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        resp.text,
    )
    if not match:
        print("Could not find __NEXT_DATA__. Site structure may have changed.")
        return []

    data = json.loads(match.group(1))
    # Navigate the JSON tree (structure may shift – inspect in browser)
    try:
        tournaments = data["props"]["pageProps"]["schedule"]["completed"]
    except KeyError:
        tournaments = []
    return tournaments


def get_leaderboard(tournament_id: str) -> pd.DataFrame:
    """Fetch leaderboard for a specific tournament."""
    url = (
        f"https://www.pgatour.com/tournaments/"
        f"{tournament_id}/leaderboard.html"
    )
    resp = polite_get(url)
    import re
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        resp.text,
    )
    if not match:
        return pd.DataFrame()

    data = json.loads(match.group(1))
    rows = []
    try:
        players = data["props"]["pageProps"]["leaderboard"]["players"]
        for p in players:
            rows.append({
                "player_id": p.get("id"),
                "name": p.get("player", {}).get("displayName"),
                "position": p.get("position", {}).get("displayValue"),
                "total_score": p.get("score", {}).get("displayValue"),
                "thru": p.get("thru", {}).get("displayValue"),
                "rounds": [
                    r.get("displayValue")
                    for r in p.get("rounds", [])
                ],
            })
    except (KeyError, TypeError):
        pass

    return pd.DataFrame(rows)


def scrape_season(year: int) -> pd.DataFrame:
    """Scrape all tournament results for a given PGA Tour season."""
    schedule = get_schedule(year)
    all_results = []
    for t in schedule:
        tid = t.get("id", "")
        print(f"  Scraping {t.get('name', tid)}...")
        df = get_leaderboard(tid)
        df["tournament"] = t.get("name")
        df["year"] = year
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    out = DATA_DIR / f"pga_results_{year}.parquet"
    combined.to_parquet(out, index=False)
    print(f"Saved {len(combined)} rows → {out}")
    return combined


if __name__ == "__main__":
    for yr in range(2020, 2026):
        scrape_season(yr)
```

---

## 2. ESPN Golf API Scraper

ESPN exposes a JSON API that doesn't require authentication.

```python
# scrapers/espn_golf.py
"""
Pull PGA Tour results from ESPN's public JSON APIs.
Coverage: ~2001–present.
"""

import pandas as pd
import requests
from shared_utils import polite_get, DATA_DIR


def get_espn_schedule(year: int) -> list[dict]:
    """Get list of PGA events for a season."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/"
        f"scoreboard?dates={year}"
    )
    resp = polite_get(url)
    events = resp.json().get("events", [])
    return [
        {"id": e["id"], "name": e["name"], "date": e["date"]}
        for e in events
    ]


def get_espn_leaderboard(event_id: str) -> pd.DataFrame:
    """Get full leaderboard for an ESPN event."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/"
        f"leaderboard?event={event_id}"
    )
    resp = polite_get(url)
    data = resp.json()

    rows = []
    try:
        competitors = data["events"][0]["competitions"][0]["competitors"]
        for c in competitors:
            athlete = c.get("athlete", {})
            rounds = c.get("linescores", [])
            rows.append({
                "player_id": athlete.get("id"),
                "name": athlete.get("displayName"),
                "position": c.get("status", {}).get("position", {}).get("displayName"),
                "total_score": c.get("score", {}).get("displayValue"),
                "total_strokes": c.get("score", {}).get("value"),
                "rounds": [r.get("displayValue") for r in rounds],
                "country": athlete.get("flag", {}).get("alt"),
            })
    except (KeyError, IndexError, TypeError):
        pass

    return pd.DataFrame(rows)


def scrape_espn_season(year: int) -> pd.DataFrame:
    """Scrape full season of ESPN PGA data."""
    events = get_espn_schedule(year)
    frames = []
    for ev in events:
        print(f"  ESPN: {ev['name']}...")
        df = get_espn_leaderboard(ev["id"])
        df["tournament"] = ev["name"]
        df["date"] = ev["date"]
        df["year"] = year
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = DATA_DIR / f"espn_pga_{year}.parquet"
    combined.to_parquet(out, index=False)
    print(f"Saved {len(combined)} rows → {out}")
    return combined


if __name__ == "__main__":
    for yr in range(2016, 2026):
        scrape_espn_season(yr)
```

---

## 3. OWGR Rankings Scraper (Selenium)

OWGR renders rankings with JavaScript, so we need Selenium.

```python
# scrapers/owgr_rankings.py
"""
Scrape Official World Golf Rankings (OWGR).
Requires Selenium because the site is JS-rendered.
"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from shared_utils import DATA_DIR
import time


def scrape_owgr(top_n: int = 200) -> pd.DataFrame:
    """Scrape current OWGR top-N players."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )

    try:
        driver.get(f"https://www.owgr.com/ranking?pageSize={top_n}")
        # Wait for table to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
        )
        time.sleep(2)  # extra buffer for JS

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        data = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 5:
                data.append({
                    "rank": cols[0].text.strip(),
                    "name": cols[2].text.strip(),
                    "country": cols[3].text.strip(),
                    "avg_points": cols[4].text.strip(),
                    "total_points": cols[5].text.strip() if len(cols) > 5 else "",
                    "events": cols[6].text.strip() if len(cols) > 6 else "",
                })
    finally:
        driver.quit()

    df = pd.DataFrame(data)
    out = DATA_DIR / "owgr_current.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} OWGR records → {out}")
    return df


if __name__ == "__main__":
    scrape_owgr(200)
```

---

## 4. Wikipedia – Historical Tournament Winners

```python
# scrapers/wiki_winners.py
"""
Scrape PGA Tour season results from Wikipedia.
Great for building a 10+ year winner dataset quickly.
"""

import pandas as pd
from bs4 import BeautifulSoup
from shared_utils import polite_get, DATA_DIR


def scrape_wiki_season(year: int) -> pd.DataFrame:
    """Scrape tournament schedule and winners for a PGA Tour season."""
    url = f"https://en.wikipedia.org/wiki/{year}_PGA_Tour"
    resp = polite_get(url)
    soup = BeautifulSoup(resp.text, "lxml")

    # Find the schedule table (usually the first large wikitable)
    tables = soup.find_all("table", {"class": "wikitable"})
    if not tables:
        print(f"No wikitable found for {year}")
        return pd.DataFrame()

    rows_data = []
    for table in tables[:2]:  # check first 2 tables
        rows = table.find_all("tr")
        headers = [h.get_text(strip=True) for h in rows[0].find_all(["th", "td"])]
        for row in rows[1:]:
            cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            if len(cols) >= 3:
                rows_data.append(dict(zip(headers, cols)))

    df = pd.DataFrame(rows_data)
    df["year"] = year
    out = DATA_DIR / f"wiki_pga_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} wiki rows → {out}")
    return df


if __name__ == "__main__":
    all_years = []
    for yr in range(2016, 2026):
        all_years.append(scrape_wiki_season(yr))

    combined = pd.concat(all_years, ignore_index=True)
    combined.to_parquet(DATA_DIR / "wiki_pga_all.parquet", index=False)
    print(f"Total: {len(combined)} rows across all years")
```

---

## 5. Weather Data (Open-Meteo)

```python
# scrapers/weather.py
"""
Fetch historical weather for tournament venues using Open-Meteo (free, no key).
"""

import pandas as pd
import requests
from shared_utils import DATA_DIR

# Course coordinates (lat, lon) – expand as needed
COURSES = {
    "Augusta National": (33.5021, -82.0227),
    "Pebble Beach": (36.5684, -121.9499),
    "TPC Sawgrass": (30.1975, -81.3959),
    "Torrey Pines": (32.8953, -117.2514),
    "Bethpage Black": (40.7474, -73.4532),
    "Pinehurst No. 2": (35.1935, -79.4692),
    "St Andrews": (56.3433, -2.8033),
    "Royal Liverpool": (53.3748, -3.1892),
    "Valhalla": (38.2553, -85.4469),
    "TPC Scottsdale": (33.6420, -111.9085),
}


def get_weather(
    lat: float, lon: float,
    start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "windspeed_10m,windgusts_10m,"
            "precipitation,cloudcover"
        ),
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    hourly = data.get("hourly", {})
    return pd.DataFrame(hourly)


def fetch_tournament_weather(
    course_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Get weather for a specific course and date range."""
    if course_name not in COURSES:
        print(f"Unknown course: {course_name}")
        return pd.DataFrame()

    lat, lon = COURSES[course_name]
    df = get_weather(lat, lon, start_date, end_date)
    df["course"] = course_name
    return df


if __name__ == "__main__":
    # Example: Masters 2024 weather
    df = fetch_tournament_weather("Augusta National", "2024-04-11", "2024-04-14")
    print(df.head())
    df.to_parquet(DATA_DIR / "weather_masters_2024.parquet", index=False)
```

---

## Best Practices

1. **Rate limiting** – Always add 1-3 second delays between requests.
2. **Caching** – Save raw HTML/JSON to disk so you don't re-scrape during development.
3. **User-Agent** – Always set a realistic browser User-Agent string.
4. **Error handling** – Sites change layouts; wrap parsing in try/except and log failures.
5. **robots.txt** – Check each site's `robots.txt` and respect disallowed paths.
6. **Parquet format** – Use Parquet over CSV for type safety, compression, and speed.
7. **Incremental scraping** – Only scrape new data; check what you already have on disk.

```python
# Example: incremental scraping pattern
from pathlib import Path

def scrape_if_missing(year: int):
    out = Path(f"data_files/pga_results_{year}.parquet")
    if out.exists():
        print(f"Skipping {year} – already scraped")
        return pd.read_parquet(out)
    return scrape_season(year)
```

---

*Next: [04_models_and_features.md](04_models_and_features.md) – ML models & feature engineering*
