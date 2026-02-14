"""
ESPN Golf API Scraper

Scrapes PGA Tour results from ESPN's public JSON APIs.
Coverage: ~2001–present.

Usage:
    python scrapers/espn_golf.py --year 2024
    python scrapers/espn_golf.py --start 2020 --end 2024
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from shared_utils import polite_get, DATA_DIR


def get_espn_schedule(year: int) -> list[dict]:
    """
    Get list of PGA events for a season from ESPN API.
    
    Args:
        year: Season year (e.g., 2024)
    
    Returns:
        List of event dictionaries with id, name, date
    """
    # ESPN's season API - this is an undocumented JSON endpoint
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/"
        f"scoreboard?dates={year}"
    )
    
    resp = polite_get(url, use_cache=True)
    data = resp.json()
    
    events = data.get("events", [])
    
    event_list = []
    for e in events:
        event_info = {
            "id": e.get("id"),
            "name": e.get("name"),
            "date": e.get("date"),
            "season": e.get("season", {}).get("year"),
        }
        event_list.append(event_info)
    
    print(f"Found {len(event_list)} events for {year}")
    return event_list


def get_espn_leaderboard(event_id: str, from_event_data: dict = None) -> pd.DataFrame:
    """
    Get full leaderboard for an ESPN event.
    
    Args:
        event_id: ESPN event ID
        from_event_data: If provided, extract from this event dict (from scoreboard)
                        instead of making a separate API call
    
    Returns:
        DataFrame with player results
    """
    # If we have event data already, use it
    if from_event_data:
        data = {"events": [from_event_data]}
    else:
        # Try the standalone scoreboard endpoint with event filter
        url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?event={event_id}"
        
        try:
            resp = polite_get(url, use_cache=True)
            data = resp.json()
        except Exception as e:
            print(f"  ⚠️  Cannot fetch leaderboard: {e}")
            return pd.DataFrame()
    
    rows = []
    try:
        # Navigate the ESPN JSON structure
        event = data.get("events", [{}])[0]
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        
        for c in competitors:
            athlete = c.get("athlete", {})
            status = c.get("status", {})
            position_obj = status.get("position", {})
            score_obj = c.get("score", {})
            
            # Handle score being either dict or string
            if isinstance(score_obj, dict):
                total_score = score_obj.get("displayValue")
                total_strokes = score_obj.get("value")
            else:
                total_score = str(score_obj) if score_obj else None
                total_strokes = None
            
            # Extract round scores
            rounds = c.get("linescores", [])
            round_scores = [r.get("displayValue") if isinstance(r, dict) else r for r in rounds]
            
            rows.append({
                "player_id": athlete.get("id"),
                "name": athlete.get("displayName"),
                "position": position_obj.get("displayName", position_obj.get("name")) if isinstance(position_obj, dict) else str(position_obj),
                "position_numeric": c.get("sortOrder"),  # Numeric position for sorting
                "total_score": total_score,
                "total_strokes": total_strokes,
                "thru": status.get("thru"),
                "rounds": round_scores,
                "country": athlete.get("flag", {}).get("alt", ""),
            })
    
    except (KeyError, IndexError, TypeError) as e:
        print(f"  ⚠️  Warning parsing leaderboard: {e}")
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        print(f"  ✓ Parsed {len(df)} players")
    else:
        print(f"  ✗ No players found")
    
    return df


def scrape_espn_season(year: int, save: bool = True) -> pd.DataFrame:
    """
    Scrape full season of ESPN PGA data.
    
    Args:
        year: Season year
        save: Whether to save to parquet file
    
    Returns:
        Combined DataFrame of all tournaments
    """
    print(f"\n{'='*60}")
    print(f"Scraping ESPN data for {year} season")
    print(f"{'='*60}\n")
    
    events = get_espn_schedule(year)
    
    if not events:
        print(f"No events found for {year}")
        return pd.DataFrame()
    
    all_frames = []
    
    for i, event in enumerate(events, 1):
        print(f"\n[{i}/{len(events)}] {event['name']} ({event['id']})...")
        
        # Try to get leaderboard - pass the event ID
        df = get_espn_leaderboard(event["id"])
        
        if not df.empty:
            # Add tournament metadata
            df["tournament"] = event["name"]
            df["tournament_id"] = event["id"]
            df["date"] = event["date"]
            df["year"] = year
            df["season"] = event.get("season", year)
            all_frames.append(df)
        else:
            print(f"  ⚠️  Skipping - no leaderboard data available")
    
    if not all_frames:
        print(f"\n⚠️  No data scraped for {year}")
        return pd.DataFrame()
    
    combined = pd.concat(all_frames, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"Summary for {year}:")
    print(f"  Tournaments with data: {combined['tournament'].nunique()}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Unique players: {combined['name'].nunique()}")
    print(f"{'='*60}\n")
    
    if save:
        out_path = DATA_DIR / f"espn_pga_{year}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"💾 Saved to: {out_path}")
    
    return combined


def scrape_multiple_seasons(start_year: int, end_year: int):
    """Scrape multiple seasons."""
    for year in range(start_year, end_year + 1):
        try:
            scrape_espn_season(year, save=True)
        except Exception as e:
            print(f"❌ Error scraping {year}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape ESPN PGA Tour data")
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
        # Single year mode
        scrape_espn_season(args.year, save=not args.no_save)
    else:
        # Range mode
        scrape_multiple_seasons(args.start, args.end)
    
    print("\n✅ Done!")
