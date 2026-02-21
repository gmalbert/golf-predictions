"""
Live Tournament Tracker

Polls ESPN's leaderboard endpoint at a configurable interval and re-runs
predictions after each round so that win probabilities stay current.

Usage
-----
    python live/tournament_tracker.py --event-id 401703511
    python live/tournament_tracker.py --event-id 401703511 --interval 15
    python live/tournament_tracker.py --list-current   # find active event IDs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── ESPN helpers ─────────────────────────────────────────────────────────────

ESPN_LEADERBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
    "?event={event_id}"
)
ESPN_SCHEDULE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
)


def get_active_event_ids() -> list[dict]:
    """Return a list of currently active (in-progress) PGA Tour events."""
    try:
        resp = requests.get(ESPN_SCHEDULE_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for ev in data.get("events", []):
            status = ev.get("status", {}).get("type", {}).get("state", "")
            events.append({
                "id": ev.get("id"),
                "name": ev.get("name"),
                "status": status,
                "active": status == "in",
            })
        return events
    except Exception as exc:
        print(f"[WARN] Could not fetch event list: {exc}")
        return []


def get_leaderboard(event_id: str) -> pd.DataFrame:
    """
    Fetch the current leaderboard for an ESPN event and return a tidy DataFrame.

    Returns an empty DataFrame if the event is not yet available.
    """
    url = ESPN_LEADERBOARD_URL.format(event_id=event_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[WARN] ESPN request failed: {exc}")
        return pd.DataFrame()

    rows = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            for entry in comp.get("competitors", []):
                athlete = entry.get("athlete", {})
                rows.append({
                    "name":        athlete.get("displayName", ""),
                    "position":    entry.get("status", {}).get("position", {}).get("displayName", ""),
                    "total_score": entry.get("score", {}).get("displayValue", "E"),
                    "thru":        entry.get("status", {}).get("thru", ""),
                    "round":       entry.get("status", {}).get("period", 0),
                })
    return pd.DataFrame(rows)


# ── Prediction refresh ───────────────────────────────────────────────────────

def _refresh_predictions(leaderboard: pd.DataFrame, tournament_name: str) -> None:
    """
    Merge current position info with player feature data and re-score.
    Prints updated top-10 win-probability ranking.
    """
    try:
        from models.predict_upcoming import predict_upcoming_tournament
        preds = predict_upcoming_tournament(tournament_name)
        if preds.empty:
            print("[INFO] No model predictions available yet.")
            return

        # Enrich with live position
        lb = leaderboard[["name", "position", "total_score", "thru"]].copy()
        lb["name_key"] = lb["name"].str.strip().str.lower()
        preds["name_key"] = preds["name"].str.strip().str.lower()
        merged = preds.merge(lb, on="name_key", how="left", suffixes=("", "_live"))

        print(
            f"\n{'Rank':<5} {'Player':<30} {'Win%':>7}  {'Pos':>5}  {'Score':>6}  {'Thru':>5}"
        )
        print("─" * 65)
        for i, row in merged.head(15).iterrows():
            print(
                f"{i+1:<5} {row['name']:<30} {row['win_probability']*100:>6.2f}%"
                f"  {str(row.get('position', '–')):>5}  "
                f"{str(row.get('total_score', '–')):>6}  "
                f"{str(row.get('thru', '–')):>5}"
            )
    except Exception as exc:
        print(f"[WARN] Could not compute predictions: {exc}")


# ── Main tracker loop ────────────────────────────────────────────────────────

def track_tournament(
    event_id: str,
    tournament_name: str = "Current PGA Tour Event",
    interval_minutes: int = 30,
    max_hours: float = 48.0,
):
    """
    Poll ESPN every `interval_minutes` and display an updated leaderboard
    with model-derived win probabilities.

    Args:
        event_id:         ESPN numeric event ID (e.g. '401703511').
        tournament_name:  Human-readable name used for model look-up.
        interval_minutes: How often to refresh (default 30 min).
        max_hours:        Stop automatically after this many hours (default 48).
    """
    max_iters = int(max_hours * 60 / interval_minutes)
    print(f"\n🏌️  Fairway Oracle – Live Tracker")
    print(f"   Event        : {tournament_name} (ID: {event_id})")
    print(f"   Refresh every: {interval_minutes} min  |  auto-stop after {max_hours}h\n")

    for iteration in range(1, max_iters + 1):
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*65}")
        print(f"  Update #{iteration}  –  {timestamp}")
        print(f"{'='*65}")

        leaderboard = get_leaderboard(event_id)
        if leaderboard.empty:
            print("  No leaderboard data yet. Tournament may not have started.")
        else:
            _refresh_predictions(leaderboard, tournament_name)

        if iteration < max_iters:
            print(f"\n  Next update in {interval_minutes} min …")
            time.sleep(interval_minutes * 60)

    print("\n[Done] Tracker stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live PGA Tour tournament tracker")
    parser.add_argument("--event-id",   default="",  help="ESPN numeric event ID")
    parser.add_argument("--name",       default="Current PGA Tour Event", help="Tournament name")
    parser.add_argument("--interval",   type=int, default=30, help="Refresh interval (minutes)")
    parser.add_argument("--max-hours",  type=float, default=48.0)
    parser.add_argument("--list-current", action="store_true",
                        help="List currently active events and exit")
    args = parser.parse_args()

    if args.list_current:
        events = get_active_event_ids()
        if not events:
            print("No active events found.")
        else:
            print(f"\n{'ID':<15} {'Status':<12} Name")
            print("─" * 60)
            for e in events:
                flag = "▶" if e["active"] else " "
                print(f"{flag} {e['id']:<14} {e['status']:<12} {e['name']}")
        sys.exit(0)

    if not args.event_id:
        parser.error("--event-id is required unless --list-current is used.")

    track_tournament(args.event_id, args.name, args.interval, args.max_hours)
