"""
Pre-generate predictions for upcoming and historical PGA tournaments and save to cache.

Run this script (or let the GitHub Action trigger it) so the Streamlit app
can serve predictions instantly from disk instead of computing them on every
user page-load.

Cache layout  (data_files/predictions_cache/):
  manifest.json                   – list of cached tournaments + timestamps
  <tournament_id>.parquet         – raw prediction DataFrame for that event
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make sure project root is on the path when executed directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data_files" / "predictions_cache"

# How many days ahead to pre-generate
DAYS_AHEAD = 60

# Maximum cache age (hours) before the GH Action considers a prediction stale
CACHE_MAX_AGE_HOURS = 23


def _tournament_slug(name: str) -> str:
    """Create a filesystem-safe slug from a tournament name."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug[:80]


def _cache_key(tournament_id, tournament_name: str) -> str:
    """Return the cache filename stem for a tournament."""
    if tournament_id:
        return str(tournament_id)
    return _tournament_slug(tournament_name)


def load_manifest() -> dict:
    manifest_path = CACHE_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def is_cache_fresh(manifest: dict, key: str, max_age_hours: int = CACHE_MAX_AGE_HOURS) -> bool:
    """Return True if the cached entry for *key* is younger than max_age_hours.

    This check only applies to *upcoming* predictions; historical entries are
    treated as always fresh once created.
    """
    entry = manifest.get(key)
    if not entry:
        return False
    if entry.get("type") != "upcoming":
        return True
    cached_at = datetime.fromisoformat(entry["cached_at"])
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
    return age_hours < max_age_hours


def pregenerate_historical(force: bool = False) -> None:
    """Pre-generate predictions for all historical tournaments found in the
    features parquet and write them to the cache.

    Avoids re-computing predictions when users browse past events.
    """
    from models.predict_tournament import predict_field

    # choose feature file as the main workflow would
    for fp_candidate in [
        ROOT / "data_files" / "espn_with_extended_features.parquet",
        ROOT / "data_files" / "espn_with_owgr_features.parquet",
        ROOT / "data_files" / "espn_player_tournament_features.parquet",
    ]:
        if fp_candidate.exists():
            features_file = fp_candidate
            break
    else:
        print("[hist] No feature parquet found, skipping historical pregeneration.")
        return

    print(f"[hist] Loading historical feature file: {features_file}")
    df_all = pd.read_parquet(features_file)

    combos = (
        df_all[['tournament', 'year']]
        .dropna()
        .drop_duplicates()
    )
    print(f"[hist] Found {len(combos)} historical tournament/year combinations.")

    manifest = load_manifest()
    updated = False

    for _, row in combos.iterrows():
        t_name = row['tournament']
        year = int(row['year'])
        key = _cache_key(None, t_name, year)
        if not force and key in manifest:
            # already computed
            continue

        print(f"  [HIST] Generating {t_name} ({year}) → key={key}")
        field = df_all[
            (df_all['tournament'] == t_name) & (df_all['year'] == year)
        ].copy()
        if field.empty:
            continue

        try:
            preds = predict_field(field)
            if preds.empty:
                continue
            parquet_path = CACHE_DIR / f"{key}.parquet"
            preds.to_parquet(parquet_path, index=False)
            manifest[key] = {
                'tournament_name': t_name,
                'tournament_id': None,
                'tournament_date': f"{year}-01-01",
                'cached_at': datetime.now(timezone.utc).isoformat(),
                'rows': len(preds),
                'type': 'historical',
                'year': year,
            }
            save_manifest(manifest)
            updated = True
        except Exception as exc:
            print(f"         → ERROR generating history: {exc}")

    if updated:
        print(f"[hist] Manifest updated with historical entries.")


def pregenerate(force: bool = False, days_ahead: int = DAYS_AHEAD) -> None:
    from models.predict_upcoming import get_upcoming_tournaments, predict_upcoming_tournament

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    # upcoming first
    print(f"[pregenerate] Fetching upcoming tournaments (next {days_ahead} days)…")
    upcoming = get_upcoming_tournaments(days_ahead=days_ahead)

    if upcoming.empty:
        print("[pregenerate] No upcoming tournaments found – nothing to cache.")
    else:
        print(f"[pregenerate] {len(upcoming)} upcoming tournament(s) found.")

        any_updated = False
        for _, row in upcoming.iterrows():
            t_name = row["name"]
            t_id = row.get("id")
            t_date = row["date"]
            key = _cache_key(t_id, t_name)
            parquet_path = CACHE_DIR / f"{key}.parquet"

            if not force and is_cache_fresh(manifest, key):
                print(f"  [SKIP] {t_name} — cache is fresh")
                continue

            print(f"  [PRED] {t_name} (id={t_id}, key={key})…")
            try:
                predictions = predict_upcoming_tournament(t_name, t_id, t_date)
                if predictions.empty:
                    print(f"         → empty predictions, skipping save")
                    continue

                # Persist all prediction columns (Streamlit enriches with odds at load time)
                predictions.to_parquet(parquet_path, index=False)

                manifest[key] = {
                    "tournament_name": t_name,
                    "tournament_id": str(t_id) if t_id else None,
                    "tournament_date": t_date.isoformat() if hasattr(t_date, "isoformat") else str(t_date),
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "rows": len(predictions),
                    "type": "upcoming",
                }
                save_manifest(manifest)
                any_updated = True
                print(f"         → saved {len(predictions)} rows to {parquet_path.name}")

            except Exception as exc:
                print(f"         → ERROR: {exc}")

        if any_updated:
            print(f"\n[pregenerate] Manifest updated: {CACHE_DIR / 'manifest.json'}")
        else:
            print("\n[pregenerate] All upcoming predictions were already fresh – nothing saved.")

    # historical next
    pregenerate_historical(force=force)


def purge_stale(days_ahead: int = DAYS_AHEAD) -> None:
    """Remove cache entries for upcoming tournaments whose date has passed.

    Historical predictions are kept indefinitely.
    """
    manifest = load_manifest()
    now = datetime.now(timezone.utc)
    to_delete = []

    for key, entry in list(manifest.items()):
        if entry.get("type") != "upcoming":
            continue
        t_date_str = entry.get("tournament_date", "")
        try:
            t_date = datetime.fromisoformat(t_date_str)
            if t_date.tzinfo is None:
                t_date = t_date.replace(tzinfo=timezone.utc)
            if t_date < now:
                to_delete.append(key)
        except Exception:
            pass

    for key in to_delete:
        parquet_path = CACHE_DIR / f"{key}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            print(f"  [PURGE] removed {parquet_path.name}")
        del manifest[key]

    if to_delete:
        save_manifest(manifest)
        print(f"[purge] Removed {len(to_delete)} stale cache entries.")
    else:
        print("[purge] No stale entries to remove.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate upcoming and historical tournament predictions.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if cache is fresh.")
    parser.add_argument("--days-ahead", type=int, default=DAYS_AHEAD, help="Look-ahead window in days.")
    parser.add_argument("--purge", action="store_true", help="Purge past-tournament cache files then exit.")
    args = parser.parse_args()

    if args.purge:
        purge_stale(days_ahead=args.days_ahead)
    else:
        purge_stale(days_ahead=args.days_ahead)   # always clean up first
        pregenerate(force=args.force, days_ahead=args.days_ahead)
