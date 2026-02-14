"""
PGA Tour Stats Scraper

Scrapes tournament results and player statistics from pgatour.com.
The PGA Tour site uses Next.js with embedded JSON data in __NEXT_DATA__ script tags.

Coverage: 2004–present (varies by stat category)

Usage:
    python scrapers/pga_tour_results.py --year 2024
    python scrapers/pga_tour_results.py --start 2020 --end 2024
"""

import argparse
import json
import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from shared_utils import polite_get, DATA_DIR


def extract_next_data(html: str) -> dict:
    """
    Extract the __NEXT_DATA__ JSON blob from a Next.js page.
    
    Args:
        html: Raw HTML content
    
    Returns:
        Parsed JSON data or empty dict if not found
    """
    # Try to find the __NEXT_DATA__ script tag
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.DOTALL
    )
    
    if not match:
        print("⚠️  Could not find __NEXT_DATA__ script tag")
        return {}
    
    try:
        data = json.loads(match.group(1))
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing __NEXT_DATA__: {e}")
        return {}


def get_pga_tour_schedule(year: int) -> list[dict]:
    """
    Get the PGA Tour schedule for a given year.
    
    Args:
        year: Season year (e.g., 2024)
    
    Returns:
        List of tournament dictionaries
    """
    # The schedule page URL
    url = f"https://www.pgatour.com/schedule/{year}"
    
    resp = polite_get(url, use_cache=True)
    data = extract_next_data(resp.text)
    
    tournaments = []
    
    try:
        # Navigation path may vary - inspect in browser to confirm
        # This is based on typical Next.js structure
        page_props = data.get("props", {}).get("pageProps", {})
        
        # Try different possible paths
        schedule_data = (
            page_props.get("schedule", {}) or
            page_props.get("tournaments", []) or
            page_props.get("completed", []) or
            []
        )
        
        if isinstance(schedule_data, dict):
            # If it's a dict, look for completed/upcoming/current
            completed = schedule_data.get("completed", [])
            upcoming = schedule_data.get("upcoming", [])
            tournaments = completed + upcoming
        elif isinstance(schedule_data, list):
            tournaments = schedule_data
        
        print(f"Found {len(tournaments)} tournaments for {year}")
        
        # Extract basic info from each tournament
        parsed = []
        for t in tournaments:
            parsed.append({
                "id": t.get("id", t.get("permNum")),
                "name": t.get("tournamentName", t.get("name")),
                "date_start": t.get("date", t.get("startDate")),
                "date_end": t.get("endDate"),
                "course": t.get("courseName", t.get("courses", [{}])[0].get("name") if t.get("courses") else None),
                "purse": t.get("purse"),
            })
        
        return parsed
        
    except (KeyError, TypeError, IndexError) as e:
        print(f"⚠️  Error parsing schedule: {e}")
        
        # Fallback: try to parse HTML directly
        print("Attempting HTML fallback parsing...")
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Look for tournament schedule table/cards
        # This is a rough fallback - structure varies
        tournament_elements = soup.find_all(class_=re.compile(r"tournament|event", re.I))
        
        fallback_tournaments = []
        for elem in tournament_elements[:50]:  # Limit to prevent over-parsing
            name = elem.get_text(strip=True)[:100]
            if name and len(name) > 3:
                fallback_tournaments.append({
                    "id": None,
                    "name": name,
                    "date_start": None,
                    "date_end": None,
                    "course": None,
                    "purse": None,
                })
        
        if fallback_tournaments:
            print(f"Fallback found {len(fallback_tournaments)} potential tournaments")
            return fallback_tournaments
        
        return []


def get_pga_tour_leaderboard(tournament_id: str, year: int = None) -> pd.DataFrame:
    """
    Get tournament leaderboard from PGA Tour.
    
    Args:
        tournament_id: PGA Tour tournament ID (e.g., "R2024014")
        year: Optional year hint
    
    Returns:
        DataFrame with player results
    """
    # Tournament leaderboard URL pattern
    if year:
        url = f"https://www.pgatour.com/tournaments/{year}/{tournament_id}/leaderboard"
    else:
        url = f"https://www.pgatour.com/leaderboard/{tournament_id}"
    
    print(f"  Fetching: {url}")
    
    resp = polite_get(url, use_cache=True)
    data = extract_next_data(resp.text)
    
    rows = []
    
    try:
        page_props = data.get("props", {}).get("pageProps", {})
        leaderboard = page_props.get("leaderboard", {})
        players = leaderboard.get("players", [])
        
        for p in players:
            player_info = p.get("player", {})
            
            # Extract round scores
            rounds = p.get("rounds", [])
            round_scores = [
                r.get("strokes", r.get("score"))
                for r in rounds
            ]
            
            rows.append({
                "player_id": player_info.get("id"),
                "name": player_info.get("displayName", player_info.get("name")),
                "position": p.get("position", {}).get("displayValue"),
                "position_numeric": p.get("position", {}).get("displayValue"),
                "total_score": p.get("total", {}).get("displayValue"),
                "total_strokes": p.get("total", {}).get("strokes"),
                "thru": p.get("thru", {}).get("displayValue"),
                "rounds": round_scores,
                "scoreToPar": p.get("scoreToPar"),
            })
        
        print(f"    ✓ Found {len(rows)} players")
        
    except (KeyError, TypeError, IndexError) as e:
        print(f"    ⚠️  Error parsing leaderboard: {e}")
    
    return pd.DataFrame(rows)


def scrape_pga_tour_season(year: int, save: bool = True) -> pd.DataFrame:
    """
    Scrape full season of PGA Tour data.
    
    Args:
        year: Season year
        save: Whether to save to parquet
    
    Returns:
        Combined DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Scraping PGA Tour data for {year} season")
    print(f"{'='*60}\n")
    
    tournaments = get_pga_tour_schedule(year)
    
    if not tournaments:
        print(f"⚠️  No tournaments found for {year}")
        return pd.DataFrame()
    
    all_frames = []
    
    for i, tourney in enumerate(tournaments, 1):
        t_id = tourney.get("id")
        t_name = tourney.get("name", "Unknown")
        
        if not t_id:
            print(f"\n[{i}/{len(tournaments)}] Skipping {t_name} (no ID)")
            continue
        
        print(f"\n[{i}/{len(tournaments)}] {t_name}")
        
        try:
            df = get_pga_tour_leaderboard(t_id, year)
            
            if not df.empty:
                df["tournament"] = t_name
                df["tournament_id"] = t_id
                df["year"] = year
                df["date"] = tourney.get("date_start")
                df["course"] = tourney.get("course")
                all_frames.append(df)
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    if not all_frames:
        print(f"\n⚠️  No leaderboards scraped for {year}")
        return pd.DataFrame()
    
    combined = pd.concat(all_frames, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Summary for {year}:")
    print(f"  Tournaments: {combined['tournament'].nunique()}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Players: {combined['name'].nunique()}")
    print(f"{'='*60}\n")
    
    if save:
        out_path = DATA_DIR / f"pga_tour_{year}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"💾 Saved to: {out_path}")
    
    return combined


def scrape_multiple_seasons(start_year: int, end_year: int):
    """Scrape multiple seasons."""
    for year in range(start_year, end_year + 1):
        try:
            scrape_pga_tour_season(year, save=True)
        except Exception as e:
            print(f"❌ Error scraping {year}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PGA Tour data")
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to scrape (e.g., 2024)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2020,
        help="Start year for range scraping (default: 2020)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2026,
        help="End year for range scraping (default: 2026)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save to parquet (testing only)"
    )
    
    args = parser.parse_args()
    
    if args.year:
        scrape_pga_tour_season(args.year, save=not args.no_save)
    else:
        scrape_multiple_seasons(args.start, args.end)
    
    print("\n✅ Done!")
