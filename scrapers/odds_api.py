"""
scrapers/odds_api.py
====================
Fetch golf outright-winner odds from The Odds API (https://the-odds-api.com).
Free tier: 500 requests/month.  We use 4 requests per full refresh (one per major).

The API only carries the 4 majors as outrights; weekly PGA Tour events are not
available on the free tier.

Usage
-----
    python scrapers/odds_api.py                 # fetch + save all markets
    python scrapers/odds_api.py --event masters # single event
    python scrapers/odds_api.py --dry-run       # show quota without fetching odds
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"
DATA_DIR = Path(__file__).parent.parent / "data_files"
DATA_DIR.mkdir(exist_ok=True)

# All golf outright markets available on free tier
GOLF_SPORTS = {
    "masters":      "golf_masters_tournament_winner",
    "pga_champ":    "golf_pga_championship_winner",
    "open":         "golf_the_open_championship_winner",
    "us_open":      "golf_us_open_winner",
}

# Bookmakers to extract (in preference order for "primary" line)
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "betrivers", "espnbet", "bovada", "betonlineag", "lowvig"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _api_key() -> str:
    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "ODDS_API_KEY not set. Add it to your .env file:\n  ODDS_API_KEY=your_key_here"
        )
    return key


def american_to_decimal(american: int) -> float:
    if american > 0:
        return round(1 + american / 100, 4)
    else:
        return round(1 + 100 / abs(american), 4)


def implied_prob(american: int) -> float:
    """Raw (vig-inclusive) implied probability from American odds."""
    if american > 0:
        return 100.0 / (american + 100.0)
    else:
        return abs(american) / (abs(american) + 100.0)


# ── Core fetch ────────────────────────────────────────────────────────────────
def fetch_event_odds(sport_key: str, regions: str = "us,us2") -> list[dict]:
    """
    Fetch outright odds for one sport/event.
    Returns raw list of event dicts from the API.

    Cost: 1 request per call.
    """
    r = requests.get(
        f"{BASE_URL}/sports/{sport_key}/odds/",
        params={
            "apiKey":      _api_key(),
            "regions":     regions,
            "markets":     "outrights",
            "oddsFormat":  "american",
            "dateFormat":  "iso",
        },
        timeout=20,
    )
    remaining   = r.headers.get("x-requests-remaining", "?")
    used        = r.headers.get("x-requests-used", "?")
    print(f"  [{r.status_code}] {sport_key}  (used={used}, remaining={remaining})")

    if not r.ok:
        print(f"  Warning: {r.status_code} {r.text[:200]}")
        return []

    return r.json()


def fetch_quota() -> dict:
    """Check remaining quota without consuming an odds request (sports list = 1 req)."""
    r = requests.get(
        f"{BASE_URL}/sports/",
        params={"apiKey": _api_key()},
        timeout=15,
    )
    return {
        "remaining": int(r.headers.get("x-requests-remaining", 0)),
        "used":      int(r.headers.get("x-requests-used", 0)),
    }


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_event_to_df(event_list: list[dict], event_label: str) -> pd.DataFrame:
    """
    Flatten a list of API event objects into a tidy DataFrame.

    Columns:
        event_label, tournament, commence_time,
        bookmaker, player, american_odds, decimal_odds, implied_prob_raw
    """
    rows = []
    for event in event_list:
        tournament    = event.get("sport_title", event_label)
        commence_time = event.get("commence_time", "")

        for book in event.get("bookmakers", []):
            book_key = book["key"]
            for market in book.get("markets", []):
                if market.get("key") != "outrights":
                    continue
                last_update = market.get("last_update", "")
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    rows.append({
                        "event_label":      event_label,
                        "tournament":       tournament,
                        "commence_time":    commence_time,
                        "bookmaker":        book_key,
                        "player":           outcome["name"],
                        "american_odds":    int(price),
                        "decimal_odds":     american_to_decimal(int(price)),
                        "implied_prob_raw": round(implied_prob(int(price)), 6),
                        "last_update":      last_update,
                    })

    df = pd.DataFrame(rows)
    return df


def build_consensus_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    From a multi-bookmaker DataFrame, build a consensus (best-available) table:
      - best_american_odds: highest American odds available across all books (best for bettor)
      - best_book: which book offers the best odds
      - dk_odds / dk_implied: DraftKings-specific line
      - market_prob_novig: no-vig normalised probability across all books (average)
    """
    if df.empty:
        return df

    # --- per-player best odds across all books ---
    best = (
        df.groupby(["event_label", "tournament", "player"])
          .apply(lambda g: g.loc[g["american_odds"].idxmax()])
          .reset_index(drop=True)
          [["event_label", "tournament", "player", "american_odds", "decimal_odds", "implied_prob_raw", "bookmaker"]]
          .rename(columns={
              "american_odds":    "best_odds",
              "decimal_odds":     "best_decimal",
              "implied_prob_raw": "best_implied_raw",
              "bookmaker":        "best_book",
          })
    )

    # --- DraftKings-specific line ---
    dk = df[df["bookmaker"] == "draftkings"][["event_label", "player", "american_odds", "implied_prob_raw"]].copy()
    dk = dk.rename(columns={"american_odds": "dk_odds", "implied_prob_raw": "dk_implied_raw"})

    # --- average no-vig prob across all books ---
    # novig for each book: normalise within that book's market
    def novig_for_book(g):
        g = g.copy()
        total = g["implied_prob_raw"].sum()
        g["novig"] = g["implied_prob_raw"] / total if total > 0 else g["implied_prob_raw"]
        return g

    df_nv = df.groupby(["event_label", "tournament", "bookmaker"], group_keys=False).apply(novig_for_book)
    avg_novig = (
        df_nv.groupby(["event_label", "tournament", "player"])["novig"]
             .mean()
             .reset_index()
             .rename(columns={"novig": "avg_novig_prob"})
    )

    # --- merge everything ---
    result = best.merge(dk, on=["event_label", "player"], how="left")
    result = result.merge(avg_novig, on=["event_label", "tournament", "player"], how="left")
    result = result.sort_values(["event_label", "avg_novig_prob"], ascending=[True, False]).reset_index(drop=True)

    return result


# ── Main fetch-all ────────────────────────────────────────────────────────────
def fetch_all_golf_odds(events: list[str] | None = None, save: bool = True) -> pd.DataFrame:
    """
    Fetch odds for all (or a subset of) golf majors.

    Args:
        events:  list of short keys from GOLF_SPORTS, e.g. ['masters', 'open']
                 If None, fetches all 4.
        save:    if True, saves raw + consensus CSVs and parquet.

    Returns:
        Consensus DataFrame (best odds + no-vig probs).
    """
    targets = {k: v for k, v in GOLF_SPORTS.items() if events is None or k in events}

    print(f"\n{'='*60}")
    print(f" Fetching golf odds – {len(targets)} event(s)")
    print(f"{'='*60}")

    all_raw: list[pd.DataFrame] = []

    for label, sport_key in targets.items():
        print(f"\n→ {label}  ({sport_key})")
        event_list = fetch_event_odds(sport_key)
        if not event_list:
            print("  No data returned.")
            continue
        df_ev = parse_event_to_df(event_list, label)
        print(f"  Parsed {len(df_ev)} lines across {df_ev['bookmaker'].nunique()} bookmakers, "
              f"{df_ev['player'].nunique()} players")
        all_raw.append(df_ev)

    if not all_raw:
        print("No odds data fetched.")
        return pd.DataFrame()

    raw_df   = pd.concat(all_raw, ignore_index=True)
    cons_df  = build_consensus_df(raw_df)

    # Timestamp string for filenames
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")

    if save:
        # Always overwrite the "latest" file, also keep a timestamped archive
        for df, stem in [(raw_df, "odds_raw"), (cons_df, "odds_consensus")]:
            df.to_parquet(DATA_DIR / f"{stem}_latest.parquet", index=False)
            df.to_csv(DATA_DIR / f"{stem}_latest.csv", index=False)
            df.to_csv(DATA_DIR / f"{stem}_{ts}.csv", index=False)
        print(f"\n✅ Saved to data_files/odds_*_latest.parquet/.csv (archive: odds_*_{ts}.csv)")

    return cons_df


# ── Odds comparison helper ────────────────────────────────────────────────────
def compare_to_model(
    model_preds: pd.DataFrame,
    event_label: str,
    name_col: str = "name",
    prob_col:  str = "win_probability",
) -> pd.DataFrame:
    """
    Join model win probabilities to consensus market odds.

    Args:
        model_preds:  DataFrame with at least name_col and prob_col columns.
        event_label:  Short key ('masters', 'pga_champ', etc.) or 'latest' to
                      load from data_files/odds_consensus_latest.parquet.
        name_col:     Column in model_preds containing player names.
        prob_col:     Column in model_preds containing model win probabilities.

    Returns:
        DataFrame sorted by edge (model_prob − market_novig_prob) descending.
    """
    cons_path = DATA_DIR / "odds_consensus_latest.parquet"
    if not cons_path.exists():
        raise FileNotFoundError("Run fetch_all_golf_odds() first to download odds.")

    cons = pd.read_parquet(cons_path)
    if event_label != "latest":
        cons = cons[cons["event_label"] == event_label]

    merged = pd.merge(
        model_preds[[name_col, prob_col]].rename(columns={name_col: "player", prob_col: "model_prob"}),
        cons[["player", "event_label", "tournament", "dk_odds", "dk_implied_raw",
              "best_odds", "best_book", "avg_novig_prob"]],
        on="player",
        how="left",
    )

    merged["edge_pp"]  = (merged["model_prob"] - merged["avg_novig_prob"]) * 100  # percentage points
    merged["edge_rel"] = merged["edge_pp"] / (merged["avg_novig_prob"].replace(0, float("nan")) * 100)

    # Half-Kelly stake suggestion (as % of bankroll)
    def half_kelly(row):
        b = row["best_odds"]
        p = row["model_prob"]
        if pd.isna(b) or pd.isna(p) or b == 0:
            return float("nan")
        dec = american_to_decimal(b)
        net = dec - 1
        k = (net * p - (1 - p)) / net
        return max(0.0, round(k * 0.5 * 100, 2))  # half-Kelly as % bankroll

    merged["half_kelly_pct"] = merged.apply(half_kelly, axis=1)

    return merged.sort_values("edge_pp", ascending=False).reset_index(drop=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch golf outright odds from The Odds API")
    parser.add_argument("--event", choices=list(GOLF_SPORTS.keys()) + ["all"],
                        default="all", help="Which event to fetch (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check quota only, don't fetch odds")
    args = parser.parse_args()

    if args.dry_run:
        quota = fetch_quota()
        print(f"Quota: {quota['remaining']} requests remaining / {quota['used']} used this month")
    else:
        events = None if args.event == "all" else [args.event]
        cons = fetch_all_golf_odds(events=events)
        if not cons.empty:
            print(f"\n{'='*60}")
            print(" CONSENSUS ODDS SUMMARY (top 10 per event by market prob)")
            print(f"{'='*60}")
            for label, grp in cons.groupby("event_label"):
                print(f"\n── {label.upper()} ──")
                print(grp[["player", "dk_odds", "best_odds", "best_book", "avg_novig_prob"]]
                      .head(10)
                      .to_string(index=False))
