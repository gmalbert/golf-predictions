"""
tools/odds_comparison.py
========================
Compare model win probabilities against market odds for golf majors.

Usage
-----
    python tools/odds_comparison.py --event masters
    python tools/odds_comparison.py --event all
    python tools/odds_comparison.py --refresh --event all   # re-fetch odds first

Output
------
    Prints a ranked table of value bets (edge = model_prob - market_novig_prob).
    Saves to models/saved_models/odds_comparison_{event}_{date}.csv
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR  = ROOT / "data_files"
MODEL_DIR = ROOT / "models" / "saved_models"


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model_predictions(event_label: str) -> pd.DataFrame:
    """
    Get model win probabilities.  For the 4 majors we run predict_upcoming;
    for 'all' we run for each major in one shot.
    """
    from models.predict_upcoming import predict_upcoming_tournament
    from scrapers.odds_api import GOLF_SPORTS

    EVENT_DATES = {
        "masters":   "2026-04-12",
        "pga_champ": "2026-05-14",
        "open":      "2026-07-16",
        "us_open":   "2026-06-18",
    }
    EVENT_NAMES = {
        "masters":   "Masters Tournament",
        "pga_champ": "PGA Championship",
        "open":      "The Open Championship",
        "us_open":   "U.S. Open",
    }

    if event_label == "all":
        frames = []
        for key in GOLF_SPORTS:
            print(f"  Running model for {key}…")
            df = predict_upcoming_tournament(
                EVENT_NAMES[key],
                tournament_date=EVENT_DATES[key],
                field_size=156,
            )
            df["event_label"] = key
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    else:
        df = predict_upcoming_tournament(
            EVENT_NAMES[event_label],
            tournament_date=EVENT_DATES[event_label],
            field_size=156,
        )
        df["event_label"] = event_label
        return df


def load_odds(event_label: str) -> pd.DataFrame:
    path = DATA_DIR / "odds_consensus_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "No odds data found. Run:  python scrapers/odds_api.py"
        )
    odds = pd.read_parquet(path)
    if event_label != "all":
        odds = odds[odds["event_label"] == event_label]
    return odds


def build_comparison(preds: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge and compute edges + half-Kelly stakes."""
    # Normalise names for merge (strip, title-case)
    preds = preds.copy()
    odds  = odds.copy()
    preds["player_key"] = preds["name"].str.strip().str.lower()
    odds["player_key"]  = odds["player"].str.strip().str.lower()

    merged = pd.merge(
        preds[["player_key", "name", "event_label", "win_probability"]],
        odds[["player_key", "event_label", "tournament", "dk_odds",
              "best_odds", "best_book", "avg_novig_prob"]],
        on=["player_key", "event_label"],
        how="left",
    ).drop(columns=["player_key"])

    merged["model_prob"]     = merged["win_probability"].astype(float)
    merged["edge_pp"]        = (merged["model_prob"] - merged["avg_novig_prob"]) * 100

    def half_kelly(row):
        b = row["best_odds"]
        p = row["model_prob"]
        if pd.isna(b) or pd.isna(p) or b == 0:
            return float("nan")
        dec = (b / 100 + 1) if b > 0 else (100 / abs(b) + 1)
        net = dec - 1
        k = (net * p - (1 - p)) / net
        return max(0.0, round(k * 0.5 * 100, 2))  # half-Kelly % of bankroll

    merged["half_kelly_pct"] = merged.apply(half_kelly, axis=1)

    return merged.sort_values("edge_pp", ascending=False).reset_index(drop=True)


def print_report(comp: pd.DataFrame, event_label: str) -> None:
    print(f"\n{'='*72}")
    print(f"  VALUE BET REPORT  -  {event_label.upper()}")
    print(f"{'='*72}")

    for ev_label, grp in comp.groupby("event_label"):
        has_odds = grp[grp["avg_novig_prob"].notna()].copy()

        tourn = grp['tournament'].iloc[0] if 'tournament' in grp.columns else ev_label
        print(f"\n-- {ev_label.upper()} ({tourn}) --")
        print(f"{'Player':<28} {'Model%':>7} {'MktNoVig%':>10} {'Edge pp':>8} {'DK Odds':>8} {'BestOdds':>9} {'Best Book':>12} {'HalfKelly%':>11}")
        print("-" * 96)

        positive = has_odds[has_odds["edge_pp"] > 0].head(15)
        if positive.empty:
            print("  No positive edges found vs market.")
        for _, r in positive.iterrows():
            dk   = f"+{int(r['dk_odds'])}" if pd.notna(r["dk_odds"]) and r["dk_odds"] > 0 else (str(int(r["dk_odds"])) if pd.notna(r["dk_odds"]) else "N/A")
            best = f"+{int(r['best_odds'])}" if pd.notna(r["best_odds"]) and r["best_odds"] > 0 else (str(int(r["best_odds"])) if pd.notna(r["best_odds"]) else "N/A")
            hk   = f"{r['half_kelly_pct']:.2f}%" if pd.notna(r["half_kelly_pct"]) else "N/A"
            print(
                f"  {r['name']:<26} "
                f"{r['model_prob']*100:>6.2f}%  "
                f"{r['avg_novig_prob']*100:>8.2f}%   "
                f"{r['edge_pp']:>+7.2f}   "
                f"{dk:>7}   "
                f"{best:>8}   "
                f"{r.get('best_book','N/A'):>11}   "
                f"{hk:>10}"
            )

        print(f"\n  -- Market/Model Divergence (model underweights) --")
        negative = has_odds[has_odds["edge_pp"] < 0].head(5)
        for _, r in negative.iterrows():
            dk   = f"+{int(r['dk_odds'])}" if pd.notna(r["dk_odds"]) and r["dk_odds"] > 0 else (str(int(r["dk_odds"])) if pd.notna(r["dk_odds"]) else "N/A")
            print(
                f"  {r['name']:<26} Model:{r['model_prob']*100:>5.2f}%  "
                f"Market:{r['avg_novig_prob']*100:>5.2f}%  "
                f"Edge:{r['edge_pp']:>+6.2f}pp  DK:{dk}"
            )


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf odds vs model value report")
    parser.add_argument("--event", choices=["masters", "pga_champ", "open", "us_open", "all"],
                        default="masters", help="Which event to analyse")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch odds from The Odds API before comparing")
    args = parser.parse_args()

    if args.refresh:
        from scrapers.odds_api import fetch_all_golf_odds, GOLF_SPORTS
        events = None if args.event == "all" else [args.event]
        fetch_all_golf_odds(events=events)

    print("\nLoading model predictions…")
    preds = load_model_predictions(args.event)
    print(f"  {len(preds)} player-rows loaded")

    print("\nLoading market odds…")
    odds = load_odds(args.event)
    print(f"  {len(odds)} player-odds loaded from data_files/odds_consensus_latest.parquet")

    comp = build_comparison(preds, odds)

    print_report(comp, args.event)

    # Save
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_path = MODEL_DIR / f"value_report_{args.event}_{ts}.csv"
    comp.to_csv(out_path, index=False)
    print(f"\n[OK] Full comparison saved to {out_path.relative_to(ROOT)}")
