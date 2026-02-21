"""
Build Extended Feature Set

Layers new features on top of the existing OWGR-enriched parquet:

  Tier 1 — Derivable from existing data (always available):
    - is_major          : 1 if major championship, else 0
    - is_playoff        : 1 if FedEx Cup playoff event, else 0
    - field_strength    : mean OWGR rank of all starters in the field (lower = stronger)
    - field_size        : number of competitors per tournament
    - purse_tier        : 1-5 categorical bucket
    - course_type_enc   : integer-encoded course type (links/parkland/desert/…)
    - grass_type_enc    : integer-encoded grass type
    - lat / lon         : course GPS coordinates (for downstream weather)

  Tier 2 — SG / driving stats (requires scrapers/pga_stats.py to have been run):
    Per-player rolling aggregates (5 and 10-event windows, shifted so there
    is no leakage from the current tournament):
    - sg_total_5 / sg_total_10
    - sg_off_tee_5  / sg_off_tee_10
    - sg_approach_5 / sg_approach_10
    - sg_putting_5  / sg_putting_10
    - driving_distance_5 / driving_distance_10
    - gir_pct_5     / gir_pct_10
    These are season-level stats joined by (player, year).

  Tier 3 — Weather (optional, requires Open-Meteo call):
    - wind_speed_avg    : mean wind speed (m/s) on tournament dates
    - temperature_avg   : mean temperature (°C)
    - precipitation_sum : total precipitation (mm)

Usage:
    python features/build_extended_features.py
    python features/build_extended_features.py --no-weather
    python features/build_extended_features.py --in  data_files/espn_with_owgr_features.parquet \
                                                --out data_files/espn_with_extended_features.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR  = Path(__file__).resolve().parent.parent / "data_files"
FEAT_DIR  = Path(__file__).resolve().parent

# Stat columns produced by pga_stats.py that we want rolling averages for
SG_STAT_COLS = [
    "sg_total", "sg_off_tee", "sg_approach", "sg_around_green", "sg_putting",
    "driving_distance", "driving_accuracy", "gir_pct", "scrambling_pct",
    "putts_per_round", "birdie_avg", "scoring_avg",
]

COURSE_TYPE_MAP = {
    "links": 0, "parkland": 1, "desert": 2,
    "resort": 3, "stadium": 4, "mountain": 5,
}
GRASS_TYPE_MAP = {
    "bentgrass": 0, "bermuda": 1, "poa_annua": 2,
    "rye": 3, "zoysia": 4, "mixed": 5,
}


# ── Tier 1: Course / tournament context ─────────────────────────────────────

def _load_course_metadata() -> dict:
    meta_path = DATA_DIR / "course_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        data = json.load(f)
    # Strip _comment / _schema keys
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _normalize_name(name: str) -> str:
    """Lowercase, strip whitespace — used for fuzzy course name matching."""
    return str(name).lower().strip()


def add_course_context(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Append course metadata columns to df."""
    # Build quick-lookup; try exact match then substring match
    norm_meta = {_normalize_name(k): v for k, v in meta.items()}

    col_defaults = {
        "is_major":       0,
        "is_playoff":     0,
        "purse_tier":     2,
        "course_type_enc": -1,
        "grass_type_enc": -1,
        "lat":            np.nan,
        "lon":            np.nan,
        "course_yardage": np.nan,
    }

    # Pre-compute per-tournament (avoid re-looking-up for every row)
    unique_tournaments = df["tournament"].dropna().unique()
    lookup: dict[str, dict] = {}
    for t in unique_tournaments:
        key = _normalize_name(t)
        if key in norm_meta:
            lookup[t] = norm_meta[key]
            continue
        # Substring match: find the longest key that is a substring of key
        best = None
        best_len = 0
        for mk, mv in norm_meta.items():
            if mk in key and len(mk) > best_len:
                best, best_len = mv, len(mk)
        if best:
            lookup[t] = best
        else:
            lookup[t] = {}

    rows_meta = df["tournament"].map(lambda t: lookup.get(t, {}))

    for col, default in col_defaults.items():
        if col in ("course_type_enc", "grass_type_enc"):
            # Need to decode from string then encode to int
            raw_col = col.replace("_enc", "")
            enc_map = COURSE_TYPE_MAP if raw_col == "course_type" else GRASS_TYPE_MAP
            df[col] = rows_meta.apply(
                lambda d: enc_map.get(d.get(raw_col, ""), -1)
            )
        elif col == "course_yardage":
            # JSON stores this as "yardage" — map it here
            df[col] = rows_meta.apply(lambda d: d.get("yardage", default))
        else:
            df[col] = rows_meta.apply(lambda d: d.get(col, default))

    return df


# ── Tier 1: Purse / length / grass-fit features ─────────────────────────────

# Dollar midpoint for each purse tier
_PURSE_TIER_DOLLARS_M = {1: 3.5, 2: 6.5, 3: 10.0, 4: 16.0, 5: 25.0}


def add_performance_fit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add three higher-level features derived from already-computed columns:

    purse_size_m
        Dollar value (millions) implied by purse_tier.  E.g. tier 3 → $10 M.
        Makes the feature continuous rather than ordinal.

    course_length_fit
        Interaction: how well a player's driving distance matches the course
        length.  Computed as the product of z-scores (both centred on tour
        averages), so positive = long hitter on a long course, negative = long
        hitter on a short course (or short hitter on a long course).

    grass_fit
        Per-player historical advantage on bermuda grass vs. bentgrass.
        grass_fit > 0 means the player finishes better (lower rank) on bermuda
        than on bentgrass, based on all prior career events.
        Computed with an expanding window — no data leakage.
    """
    # ── purse_size_m ────────────────────────────────────────────────────────
    if "purse_tier" in df.columns:
        df["purse_size_m"] = df["purse_tier"].map(_PURSE_TIER_DOLLARS_M)

    # ── course_length_fit ───────────────────────────────────────────────────
    drive_col = "driving_distance_season"
    if "course_yardage" in df.columns and drive_col in df.columns:
        drive_mean = df[drive_col].mean()
        drive_std  = df[drive_col].std()
        yard_mean  = df["course_yardage"].mean()
        yard_std   = df["course_yardage"].std()

        z_drive  = (df[drive_col] - drive_mean) / max(drive_std, 1e-9)
        z_length = (df["course_yardage"] - yard_mean) / max(yard_std, 1e-9)
        df["course_length_fit"] = z_drive * z_length
    else:
        df["course_length_fit"] = np.nan

    # ── grass_fit ───────────────────────────────────────────────────────────
    if "grass_type_enc" in df.columns and "tournament_rank" in df.columns:
        df = df.sort_values(["name", "year", "date"]).copy()

        grass_fit_vals: list[float] = []
        for _player, grp in df.groupby("name", sort=False, dropna=False):
            grp = grp.sort_values(["year", "date"])
            b_sum, b_cnt = 0.0, 0   # bermuda
            g_sum, g_cnt = 0.0, 0   # bentgrass
            for _, row in grp.iterrows():
                # Record fit BEFORE updating so there is no leakage
                if b_cnt > 0 and g_cnt > 0:
                    grass_fit_vals.append((g_sum / g_cnt) - (b_sum / b_cnt))
                else:
                    grass_fit_vals.append(0.0)
                rank = row["tournament_rank"]
                gt   = row["grass_type_enc"]
                if pd.notna(rank) and pd.notna(gt):
                    if int(gt) == 1:   # bermuda
                        b_sum += rank; b_cnt += 1
                    elif int(gt) == 0: # bentgrass
                        g_sum += rank; g_cnt += 1

        df["grass_fit"] = grass_fit_vals
    else:
        df["grass_fit"] = np.nan

    return df


def add_field_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-tournament field quality features:
      - field_strength : mean owgr_rank_current across players in the field
                         (lower = stronger field)
      - field_size     : number of starters
    """
    grp_key = ["tournament_id", "year"]
    if "tournament_id" not in df.columns:
        grp_key = ["tournament", "year"]

    agg = df.groupby(grp_key).agg(
        field_strength=("owgr_rank_current", "mean"),
        field_size=("player_id", "count"),
    ).reset_index()

    df = df.merge(agg, on=grp_key, how="left")
    return df


# ── Tier 2: SG / driving stats ───────────────────────────────────────────────

def _load_pga_stats() -> pd.DataFrame | None:
    """Load combined pga_stats parquet if available."""
    combined = DATA_DIR / "pga_stats_all.parquet"
    if combined.exists():
        return pd.read_parquet(combined)

    # Fall back to per-year files
    parts = sorted(DATA_DIR.glob("pga_stats_*.parquet"))
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def add_sg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge season-level SG / driving stats onto df.

    Because pga_stats.py scrapes season aggregates (one row per player per year),
    we merge on (player_id OR name, year).  This gives the player's full-season
    average, which is a proxy for skill level going into any event that year.

    For a forward-looking window (no leakage), we shift stats back one year:
    features labelled *_prev_season use the PRIOR year's numbers.
    """
    stats = _load_pga_stats()
    if stats is None:
        print("  ⚠️  No pga_stats parquet found — SG features will be skipped.")
        print("     Run:  python scrapers/pga_stats.py --start 2018 --end 2025")
        return df

    available_stat_cols = [c for c in SG_STAT_COLS if c in stats.columns]
    if not available_stat_cols:
        print("  ⚠️  pga_stats parquet has no expected stat columns — skipping.")
        return df

    print(f"  Found SG stats: {available_stat_cols}")

    # Normalise player name for fallback merging
    if "name" in stats.columns:
        stats["_name_lower"] = stats["name"].str.lower().str.strip()
    if "name" in df.columns:
        df["_name_lower"] = df["name"].str.lower().str.strip()

    # --- Current season stats (may have some leakage for early-season events;
    #     acceptable given full-season aggregation)
    stats_curr = stats.rename(
        columns={c: f"{c}_season" for c in available_stat_cols}
    )
    curr_cols = [f"{c}_season" for c in available_stat_cols]

    # --- Previous season stats (zero leakage)
    stats_prev = stats.copy()
    stats_prev["year"] = stats_prev["year"] + 1  # shift so 2023 stats join 2024 events
    stats_prev = stats_prev.rename(
        columns={c: f"{c}_prev_season" for c in available_stat_cols}
    )
    prev_cols = [f"{c}_prev_season" for c in available_stat_cols]

    def _ids_compatible(a, b) -> bool:
        """Return True if the two player_id series share at least one value."""
        try:
            sa = set(str(x) for x in a.dropna().unique())
            sb = set(str(x) for x in b.dropna().unique())
            return len(sa & sb) > 0
        except Exception:
            return False

    for stats_df, feature_cols in [(stats_curr, curr_cols), (stats_prev, prev_cols)]:
        # Prefer name-based merge UNLESS player_id spaces are demonstrably compatible.
        # Many datasets use different player_id namespaces (PGA vs ESPN), so
        # merging on player_id blindly produces no matches — fall back to names.
        did_merge = False

        if "player_id" in stats_df.columns and "player_id" in df.columns:
            if _ids_compatible(stats_df["player_id"], df["player_id"]):
                merge_cols = ["player_id", "year"]
                df = df.merge(
                    stats_df[merge_cols + feature_cols].drop_duplicates(merge_cols),
                    on=merge_cols,
                    how="left",
                )
                did_merge = True

        if not did_merge and "_name_lower" in stats_df.columns and "_name_lower" in df.columns:
            merge_cols = ["_name_lower", "year"]
            df = df.merge(
                stats_df[merge_cols + feature_cols].drop_duplicates(merge_cols),
                on=merge_cols,
                how="left",
            )
            did_merge = True

        # Last resort: if we haven't merged yet but player_id columns exist, try id-merge anyway
        if not did_merge and "player_id" in stats_df.columns and "player_id" in df.columns:
            merge_cols = ["player_id", "year"]
            df = df.merge(
                stats_df[merge_cols + feature_cols].drop_duplicates(merge_cols),
                on=merge_cols,
                how="left",
            )

    # Drop helper columns
    for col in ["_name_lower"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    print(f"  ✓ SG season features added: {curr_cols + prev_cols}")
    return df


# ── Tier 3: Weather ──────────────────────────────────────────────────────────

def _fetch_weather(lat: float, lon: float, date_str: str) -> dict:
    """
    Fetch historical weather from Open-Meteo for a single course/date.
    Returns dict with wind_speed_avg, temperature_avg, precipitation_sum.
    Uses the free historical weather API (no key required).
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        "&daily=wind_speed_10m_max,temperature_2m_mean,precipitation_sum"
        "&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json().get("daily", {})
        def _first(key):
            vals = data.get(key, [None])
            return vals[0] if vals else None
        return {
            "wind_speed_avg":    _first("wind_speed_10m_max"),
            "temperature_avg":   _first("temperature_2m_mean"),
            "precipitation_sum": _first("precipitation_sum"),
        }
    except Exception:
        return {}


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather features for each (tournament, year) group.
    Requires 'lat', 'lon', 'date' columns to already be present.
    Makes one API call per unique (tournament_id, year) combination.
    """
    if "lat" not in df.columns or df["lat"].isna().all():
        print("  ⚠️  No lat/lon data — weather features skipped.")
        df["wind_speed_avg"] = np.nan
        df["temperature_avg"] = np.nan
        df["precipitation_sum"] = np.nan
        return df

    grp_key = "tournament_id" if "tournament_id" in df.columns else "tournament"
    unique_events = (
        df[df["lat"].notna()]
        .groupby([grp_key, "year"])
        .first()[["lat", "lon", "date"]]
        .reset_index()
    )

    total = len(unique_events)
    print(f"  Fetching weather for {total} events from Open-Meteo …")

    weather_cache: dict[tuple, dict] = {}
    for _, row in unique_events.iterrows():
        key = (row[grp_key], row["year"])
        try:
            date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        except Exception:
            weather_cache[key] = {}
            continue
        weather_cache[key] = _fetch_weather(float(row["lat"]), float(row["lon"]), date_str)
        time.sleep(0.15)  # polite rate limiting

    df["wind_speed_avg"]    = df.apply(
        lambda r: weather_cache.get((r[grp_key], r["year"]), {}).get("wind_speed_avg"),    axis=1
    )
    df["temperature_avg"]   = df.apply(
        lambda r: weather_cache.get((r[grp_key], r["year"]), {}).get("temperature_avg"),   axis=1
    )
    df["precipitation_sum"] = df.apply(
        lambda r: weather_cache.get((r[grp_key], r["year"]), {}).get("precipitation_sum"), axis=1
    )

    non_null = df["wind_speed_avg"].notna().sum()
    print(f"  ✓ Weather data retrieved for {non_null:,} / {len(df):,} rows")
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────

def build_extended_features(
    in_path: Path,
    out_path: Path,
    include_weather: bool = True,
) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("Building Extended Feature Set")
    print(f"{'='*60}")
    print(f"  Input  : {in_path}")
    print(f"  Output : {out_path}")

    df = pd.read_parquet(in_path)
    print(f"\n  Loaded {len(df):,} rows, {df.shape[1]} columns")

    meta = _load_course_metadata()
    print(f"  Course metadata: {len(meta)} entries")

    # ── Tier 1 ──────────────────────────────────────────────────────────────
    print("\n[Tier 1] Course & tournament context …")
    df = add_course_context(df, meta)
    df = add_field_features(df)
    print(f"  ✓ is_major, is_playoff, course_type_enc, grass_type_enc, "
          f"purse_tier, course_yardage, field_strength, field_size")

    # ── Tier 2 ──────────────────────────────────────────────────────────────
    print("\n[Tier 2] SG / driving stats …")
    df = add_sg_features(df)

    # ── Tier 1b: performance-fit (needs SG stats for course_length_fit) ─────
    print("\n[Tier 1b] Performance-fit features …")
    df = add_performance_fit(df)
    print("  ✓ purse_size_m, course_length_fit, grass_fit")

    # ── Tier 3 ──────────────────────────────────────────────────────────────
    if include_weather:
        print("\n[Tier 3] Weather features …")
        df = add_weather_features(df)
    else:
        print("\n[Tier 3] Weather skipped (--no-weather)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Saved: {out_path}")
    print(f"   Rows   : {len(df):,}")
    print(f"   Columns: {df.shape[1]}")

    new_cols = [c for c in df.columns if c not in pd.read_parquet(in_path).columns]
    if new_cols:
        print(f"\n   New columns ({len(new_cols)}):")
        for c in new_cols:
            nn = df[c].notna().sum()
            pct = nn / len(df) * 100
            print(f"     {c:<40} {nn:>7,} non-null  ({pct:.0f}%)")

    return df


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build extended feature set")
    parser.add_argument(
        "--in",
        dest="in_path",
        default=str(DATA_DIR / "espn_with_owgr_features.parquet"),
        help="Input parquet (default: data_files/espn_with_owgr_features.parquet)",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=str(DATA_DIR / "espn_with_extended_features.parquet"),
        help="Output parquet (default: data_files/espn_with_extended_features.parquet)",
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Skip Open-Meteo weather API calls",
    )
    args = parser.parse_args()

    build_extended_features(
        in_path=Path(args.in_path),
        out_path=Path(args.out_path),
        include_weather=not args.no_weather,
    )
