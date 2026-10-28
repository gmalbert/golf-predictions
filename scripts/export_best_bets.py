"""Publish dated model predictions matched to fresh quotes for Sports Picks Grid."""
from __future__ import annotations
import csv
import json
import math
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_files"
MODEL_VERSION = "1.0.0"
EV_THRESHOLD = .03


def _tier_from_edge(edge):
    return "Elite" if edge >= .06 else "Strong" if edge >= .03 else "Good" if edge >= .01 else "Standard"


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _key(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\s+(tournament\s+)?winner$", "", value.lower()))


def _write(bets, notes="", *, status=None, source_generated_at=None):
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {
        "sport": "Golf", "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION, "tier_definition": "edge-v1",
        "season": str(date.today().year), "status": status or ("ok" if bets else "no_picks"),
        "source_generated_at": source_generated_at, "notes": notes,
    }, "bets": bets}
    target = DATA / "best_bets_today.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(target)
    print(f"[Golf] {payload['meta']['status']}: {len(bets)} picks")


def export(for_date=None, now=None):
    import pandas as pd
    now = now or datetime.now(timezone.utc)
    today = for_date or now.date()
    parquet = DATA / "odds_consensus_all.parquet"
    if not parquet.exists():
        parquet = DATA / "odds_consensus_latest.parquet"
    csv_path = DATA / "odds_consensus_latest.csv"
    try:
        if parquet.exists():
            quotes = pd.read_parquet(parquet).to_dict("records")
        elif csv_path.exists():
            with csv_path.open(encoding="utf-8", newline="") as stream:
                quotes = list(csv.DictReader(stream))
        else:
            _write([], "Market odds artifact is missing", status="pipeline_pending")
            return
    except Exception as exc:
        _write([], f"Cannot read odds artifact: {type(exc).__name__}", status="pipeline_failed")
        raise
    fresh = []
    for quote in quotes:
        observed = pd.to_datetime(quote.get("fetched_at"), errors="coerce", utc=True)
        if pd.notna(observed) and 0 <= (now - observed.to_pydatetime()).total_seconds() <= 36 * 3600:
            fresh.append(quote)
    if not fresh:
        _write([], "No timestamped market quotes within 36 hours; old consensus CSVs cannot be relabelled as current", status="pipeline_pending")
        return

    predictions = []
    failures = []
    for source in sorted((DATA / "predictions_cache").glob("*.parquet")):
        try:
            frame = pd.read_parquet(source)
            if not {"name", "tournament", "date", "win_probability"}.issubset(frame.columns):
                continue
            dates = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.date
            frame = frame[(dates >= today) & (dates <= today + timedelta(days=7))]
            predictions.extend(frame.to_dict("records"))
        except Exception:
            failures.append(source.name)
    if not predictions:
        _write([], "No dated model predictions in the next seven days" + ("; unreadable caches: " + ", ".join(failures) if failures else ""), status="pipeline_failed" if failures else "pipeline_pending")
        return

    bets, matched = [], 0
    seen = set()
    for model in predictions:
        for quote in fresh:
            event = quote.get("event_name") or quote.get("tournament")
            if _key(model.get("tournament")) != _key(event):
                continue
            if not _key(model.get("name")) or _key(model["name"]) != _key(quote.get("player_name") or quote.get("player")):
                continue
            probability, odds = _number(model.get("win_probability")), _number(quote.get("best_odds"))
            if probability is None or not 0 <= probability <= 1 or odds is None or abs(odds) < 100:
                continue
            matched += 1
            implied = abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
            edge = round(probability - implied, 4)
            if edge < EV_THRESHOLD:
                continue
            game_date = pd.Timestamp(model["date"]).date().isoformat()
            identity = (game_date, _key(event), _key(model["name"]))
            if identity in seen:
                continue
            seen.add(identity)
            bets.append({
                "game_date": game_date, "game": model["tournament"], "game_time": None,
                "bet_type": "tournament_winner", "pick": model["name"],
                "confidence": round(probability, 6), "edge": edge, "odds": int(odds),
                "tier": _tier_from_edge(edge), "market_probability": round(implied, 6),
                "quote_observed_at": quote["fetched_at"],
                "notes": f"Book: {quote.get('best_book', '')}; matched dated model forecast",
            })
    bets.sort(key=lambda bet: bet["edge"], reverse=True)
    _write(bets, "No model/market pairs for the same upcoming event and player" if not matched else "",
           status=None if matched else "pipeline_pending",
           source_generated_at=min(str(q["fetched_at"]) for q in fresh))


if __name__ == "__main__":
    export()
