"""
scrapers/rotowire_odds.py
-------------------------
Scrapes outright win odds for PGA Tour events from RotoWire's betting page.

Endpoint discovered:
  https://www.rotowire.com/betting/golf/tables/golf-event.php?event=<eventID>

Returns JSON with 72+ players and odds from up to 7 books:
  mgm, betrivers, draftkings, fanduel, caesars, hardrock, thescore

Usage:
  python scrapers/rotowire_odds.py                  # fetch current event
  python scrapers/rotowire_odds.py --all            # fetch all listed events
  python scrapers/rotowire_odds.py --event 5008412  # fetch specific event ID
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://www.rotowire.com"
GOLF_BETTING_PAGE = f"{BASE_URL}/betting/golf/"
ODDS_TABLE_URL = f"{BASE_URL}/betting/golf/tables/golf-event.php"

BOOKS = ["mgm", "betrivers", "draftkings", "fanduel", "caesars", "hardrock", "thescore"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": GOLF_BETTING_PAGE,
}

DATA_DIR = Path(__file__).parent.parent / "data_files"


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def get_event_list() -> list[dict]:
    """
    Scrape the RotoWire golf betting hub and return all listed events.

    Returns a list of dicts:
        [{"name": "The Genesis Invitational", "event_id": "5008412", "is_current": True}, ...]

    The current (default) event has event_id extracted from the inline JS.
    Upcoming events have event_ids from the <option data-url> elements.
    """
    r = requests.get(GOLF_BETTING_PAGE, headers={**HEADERS, "Accept": "text/html"}, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Current event ID lives in the inline JS: const eventID = "XXXXXXX";
    current_id = None
    inline_scripts = [s.string for s in soup.find_all("script", src=False) if s.string]
    for script in inline_scripts:
        m = re.search(r'const\s+eventID\s*=\s*"(\d+)"', script)
        if m:
            current_id = m.group(1)
            break

    # Event options in the selector
    options = soup.find_all("option", attrs={"data-url": True})
    events = []
    for opt in options:
        label = opt.get_text(strip=True)
        data_url = opt["data-url"]
        m = re.search(r"\?event=(\d+)", data_url)
        if m:
            event_id = m.group(1)
            is_current = False
        else:
            # The current event has data-url=/betting/golf/ (no ?event=)
            event_id = current_id
            is_current = True
        events.append({"name": label, "event_id": event_id, "is_current": is_current})

    return events


# ---------------------------------------------------------------------------
# Odds fetcher
# ---------------------------------------------------------------------------

def fetch_event_odds(event_id: str) -> list[dict]:
    """
    Fetch the raw odds JSON from RotoWire for a given event ID.
    Returns a list of player dicts.
    """
    r = requests.get(ODDS_TABLE_URL, params={"event": event_id}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Data builder
# ---------------------------------------------------------------------------

def build_consensus_df(raw: list[dict], event_name: str, event_id: str) -> pd.DataFrame:
    """
    Convert raw RotoWire player odds into the same consensus format used by
    odds_consensus_latest.parquet (for compatibility with enrich_predictions_with_odds).

    Output columns:
        player_name, event_label, event_name, event_id,
        dk_odds,            -- DraftKings American odds (int or NaN)
        best_odds,          -- Best American odds across all books (int)
        best_book,          -- Book offering best_odds
        avg_novig_prob,     -- Average no-vig win probability across available books
        fetched_at
    """
    rows = []
    for p in raw:
        # Collect all win odds across books (American format as strings like "300", "-150")
        book_odds = {}
        for book in BOOKS:
            val = p.get(f"{book}_win")
            if val is not None:
                try:
                    book_odds[book] = int(val)
                except (ValueError, TypeError):
                    pass

        if not book_odds:
            continue

        # Convert American odds to implied probability (no-vig)
        def american_to_prob(odds: int) -> float:
            if odds > 0:
                return 100 / (odds + 100)
            else:
                return -odds / (-odds + 100)

        implied_probs = {b: american_to_prob(o) for b, o in book_odds.items()}

        # No-vig: normalise across all books' implied probs
        avg_implied = sum(implied_probs.values()) / len(implied_probs)

        # Best odds = highest American odds (most +ve, i.e., best payout)
        best_book = max(book_odds, key=lambda b: book_odds[b])
        best_odds_val = book_odds[best_book]

        dk_odds_val = book_odds.get("draftkings", None)

        rows.append({
            "player_name": p["name"],
            "event_id": event_id,
            "event_name": event_name,
            "dk_odds": dk_odds_val,
            "best_odds": best_odds_val,
            "best_book": best_book,
            "avg_novig_prob": round(avg_implied, 6),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_consensus(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save consensus DataFrame as parquet, mirroring odds_consensus_latest.parquet format."""
    if path is None:
        path = DATA_DIR / "odds_consensus_latest.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} players to {path}")
    return path


def save_event_snapshot(df: pd.DataFrame, event_id: str) -> Path:
    """Also save a timestamped copy per event for historical reference."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = DATA_DIR / "odds_history" / f"rotowire_{event_id}_{ts}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Snapshot saved to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scrape PGA Tour win odds from RotoWire")
    parser.add_argument("--event", type=str, default=None, help="Specific RotoWire event ID")
    parser.add_argument("--all", action="store_true", help="Fetch all listed events")
    parser.add_argument("--list", action="store_true", help="List available events and exit")
    parser.add_argument("--no-save", action="store_true", help="Print results only, don't save")
    args = parser.parse_args()

    events = get_event_list()
    print(f"Events on RotoWire ({len(events)}):")
    for e in events:
        marker = " <-- current" if e["is_current"] else ""
        print(f"  [{e['event_id']}] {e['name']}{marker}")

    if args.list:
        return

    # Determine which events to fetch
    if args.event:
        targets = [e for e in events if e["event_id"] == args.event]
        if not targets:
            # Still try even if not in the list
            targets = [{"name": f"Event {args.event}", "event_id": args.event, "is_current": False}]
    elif args.all:
        targets = events
    else:
        # Default: current event only
        targets = [e for e in events if e["is_current"]]
        if not targets:
            targets = [events[0]]

    all_dfs = []
    for ev in targets:
        print(f"\nFetching odds for: {ev['name']} (ID={ev['event_id']})...")
        try:
            raw = fetch_event_odds(ev["event_id"])
            df = build_consensus_df(raw, ev["name"], ev["event_id"])
            print(f"  {len(df)} players with win lines")

            # Show top 10 favourites
            top = df.nlargest(10, "avg_novig_prob").copy()
            top["dk_fmt"] = top["dk_odds"].apply(
                lambda x: f"+{int(x)}" if pd.notna(x) and x >= 0 else (f"{int(x)}" if pd.notna(x) else "--")
            )
            top["best_fmt"] = top["best_odds"].apply(lambda x: f"+{x}" if x >= 0 else str(x))
            print(f"  {'Player':<25} {'DK':>8} {'Best':>8} {'Book':<12} {'NoVig%':>8}")
            print("  " + "-" * 68)
            for _, row in top.iterrows():
                print(
                    f"  {row['player_name']:<25} {row['dk_fmt']:>8} {row['best_fmt']:>8} "
                    f"{row['best_book']:<12} {row['avg_novig_prob']*100:>7.2f}%"
                )

            all_dfs.append(df)

            if not args.no_save:
                save_event_snapshot(df, ev["event_id"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    if all_dfs and not args.no_save:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Save the first/current event as the main consensus file
        save_consensus(all_dfs[0])
        print(f"\nDone. {len(all_dfs)} event(s) fetched.")


if __name__ == "__main__":
    main()
