"""
Export daily best bets for the Sports Picks Grid aggregator.

Reads data_files/odds_consensus_latest.csv which contains per-player
tournament winner odds. Computes edge as (avg_novig_prob - dk_implied_raw)
and writes data_files/best_bets_today.json in the unified schema.

Usage:
    python scripts/export_best_bets.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_files"

SPORT = "Golf"
MODEL_VERSION = "1.0.0"
EV_THRESHOLD = 0.03  # minimum edge to include

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_from_edge(edge: float) -> str:
    if edge >= 0.08:
        return "Elite"
    if edge >= 0.04:
        return "Strong"
    if edge >= 0.03:
        return "Good"
    return "Standard"


def _american_from_decimal(dec: float) -> int | None:
    """Convert decimal odds to American integer."""
    try:
        dec = float(dec)
        if dec <= 0:
            return None
        if dec >= 2.0:
            return int(round((dec - 1) * 100))
        return int(round(-100 / (dec - 1)))
    except Exception:
        return None


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export() -> None:
    import csv

    odds_path = DATA / "odds_consensus_latest.csv"
    if not odds_path.exists():
        print(f"[golf export] {odds_path} not found — writing empty bets")
        _write([], "odds_consensus_latest.csv not found")
        return

    bets: list[dict] = []
    season = str(date.today().year)

    with odds_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tournament = row.get("tournament", "").strip()
            player = row.get("player", "").strip()
            if not tournament or not player:
                continue

            avg_novig = _safe_float(row.get("avg_novig_prob", 0))
            dk_implied = _safe_float(row.get("dk_implied_raw", 0))
            dk_odds_raw = row.get("dk_odds", "")
            dk_dec = _safe_float(row.get("best_decimal", 0))

            # Prefer DK implied; fallback to best_implied_raw
            market_implied = dk_implied if dk_implied > 0 else _safe_float(row.get("best_implied_raw", 0))

            if avg_novig <= 0 or market_implied <= 0:
                continue

            edge = avg_novig - market_implied

            if edge < EV_THRESHOLD:
                continue

            # Build American odds from dk_odds column (already American) or decimal fallback
            try:
                dk_odds_int = int(float(dk_odds_raw)) if dk_odds_raw else None
            except Exception:
                dk_odds_int = _american_from_decimal(dk_dec) if dk_dec > 0 else None

            confidence = round(min(max(avg_novig, 0.0), 1.0), 4)
            edge_val = round(edge, 4)
            tier = _tier_from_edge(edge_val)

            bets.append(
                {
                    "game_date": str(date.today()),
                    "game": tournament,
                    "game_time": None,
                    "bet_type": "tournament_winner",
                    "pick": player,
                    "confidence": confidence,
                    "edge": edge_val,
                    "odds": dk_odds_int,
                    "tier": tier,
                    "notes": (
                        f"Avg no-vig: {avg_novig:.1%}  |  "
                        f"Market implied: {market_implied:.1%}  |  "
                        f"Book: {row.get('best_book','')}"
                    ),
                }
            )

    # Sort by edge descending
    bets.sort(key=lambda b: b["edge"], reverse=True)
    _write(bets)


def _write(bets: list[dict], notes: str = "") -> None:
    payload = {
        "meta": {
            "sport": SPORT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
            "season": str(date.today().year),
            "notes": notes,
        },
        "bets": bets,
    }
    out = DATA / "best_bets_today.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[golf export] Wrote {len(bets)} bets → {out}")


if __name__ == "__main__":
    export()
