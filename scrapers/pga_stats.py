"""
PGA Tour Stats Scraper

Scrapes per-season player statistics from pgatour.com stats pages.
These stats are NOT available in the ESPN leaderboard API — they must be
scraped from the PGA Tour stats portal.

Available stat IDs (hard-coded, stable):
  - Strokes Gained: Total        02675
  - Strokes Gained: Off the Tee  02567
  - Strokes Gained: Approach     02568
  - Strokes Gained: Around Green 02569
  - Strokes Gained: Putting      02564
  - Driving Distance             101
  - Driving Accuracy             102
  - Greens in Regulation         103
  - Scrambling                   130
  - Putts per Round              104
  - Birdie Average               156
  - Scoring Average              120

Usage:
    python scrapers/pga_stats.py --year 2024
    python scrapers/pga_stats.py --start 2018 --end 2024
    python scrapers/pga_stats.py --year 2024 --stat sg_total
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils import polite_get, DATA_DIR

# ── Stat ID Catalogue ────────────────────────────────────────────────────────
# Source: pgatour.com/stats/detail/{id} — stable across seasons
STAT_IDS: dict[str, str] = {
    "sg_total":          "02675",
    "sg_off_tee":        "02567",
    "sg_approach":       "02568",
    "sg_around_green":   "02569",
    "sg_putting":        "02564",
    "driving_distance":  "101",
    "driving_accuracy":  "102",
    "gir_pct":           "103",
    "scrambling_pct":    "130",
    "putts_per_round":   "104",
    "birdie_avg":        "156",
    "scoring_avg":       "120",
}

STAT_GROUPS = {
    "sg":      ["sg_total", "sg_off_tee", "sg_approach", "sg_around_green", "sg_putting"],
    "driving": ["driving_distance", "driving_accuracy"],
    "greens":  ["gir_pct", "scrambling_pct"],
    "putting": ["putts_per_round"],
    "scoring": ["birdie_avg", "scoring_avg"],
    "all":     list(STAT_IDS.keys()),
}

# ── Core scraping functions ──────────────────────────────────────────────────

def _extract_next_data(html: str) -> dict:
    """Extract the __NEXT_DATA__ JSON blob from a Next.js page."""
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html, re.DOTALL
    )
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _parse_stat_table_html(html: str, stat_name: str) -> pd.DataFrame:
    """
    Fallback: parse the stats HTML table using BeautifulSoup.
    Used when __NEXT_DATA__ doesn't have player-level rows.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_=re.compile(r"Table|stats", re.I))
    if table is None:
        # Try any table
        table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    rows = []
    headers = []
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]

    tbody = table.find("tbody")
    if tbody is None:
        return pd.DataFrame()

    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if headers else None)
    df[stat_name] = pd.to_numeric(df.get(df.columns[-1], pd.Series()), errors="coerce")
    return df


def scrape_stat(stat_name: str, year: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Scrape a single stat for a given season.

    Returns a DataFrame with columns:
        player_id, name, rank, {stat_name}, year, stat_name_key

    Data source: https://www.pgatour.com/stats/detail/{stat_id}?year={year}
    The PGA Tour stats pages expose data via several mechanisms:
      1. __NEXT_DATA__ JSON blob (preferred)
      2. GraphQL / XHR API fallback
      3. HTML table fallback
    """
    stat_id = STAT_IDS.get(stat_name)
    if stat_id is None:
        raise ValueError(f"Unknown stat '{stat_name}'. Valid: {list(STAT_IDS)}")

    url = f"https://www.pgatour.com/stats/detail/{stat_id}?year={year}&country=ANY"
    print(f"  [{stat_name}] Fetching: {url}")

    try:
        resp = polite_get(url, use_cache=use_cache)
    except Exception as e:
        print(f"  ⚠️  Request failed for {stat_name}/{year}: {e}")
        return pd.DataFrame()

    rows = []

    # --- Attempt 1: __NEXT_DATA__ ---
    next_data = _extract_next_data(resp.text)
    if next_data:
        try:
            page_props = next_data.get("props", {}).get("pageProps", {})

            # Multiple possible paths in the JSON tree
            stat_data = (
                page_props.get("statDetails", {})
                or page_props.get("stat", {})
                or {}
            )

            # Try rows directly
            player_rows = (
                stat_data.get("rows", [])
                or stat_data.get("playerStats", [])
                or stat_data.get("players", [])
                or []
            )

            # Fallback: some pages put StatDetails inside the dehydratedState
            if not player_rows:
                ds = page_props.get("dehydratedState") or page_props.get("dehydrated") or {}
                try:
                    # dehydratedState can be shaped two ways: {"queries": [...]} or {"state": {"queries": [...]}}
                    queries = []
                    if isinstance(ds, dict):
                        queries = ds.get("queries", []) or ds.get("state", {}).get("queries", [])

                    for q in queries:
                        # Some entries are {dehydratedAt, state: { data: ... }, queryKey }
                        data = q.get("state", {}).get("data") if isinstance(q, dict) else None
                        if isinstance(data, dict) and data.get("__typename") == "StatDetails":
                            player_rows = (data.get("rows", []) or data.get("playerStats", []) or [])
                            break
                except Exception:
                    player_rows = player_rows or []

            for p in player_rows:
                # player identifier — prefer PGA plrId / playerId, fall back to any id field
                player_id = p.get("plrId") or p.get("playerId") or p.get("id")

                # player name — pages use several keys (playerName, plrName, name, displayName)
                player_name = (
                    p.get("playerName")
                    or p.get("plrName")
                    or p.get("name")
                    or p.get("displayName")
                )

                # stat value: many payloads put numeric value inside a "stats" array
                stat_val = (
                    p.get("statValue")
                    or p.get("value")
                    or p.get("avg")
                    or p.get("stat")
                )
                if stat_val is None and isinstance(p.get("stats"), list):
                    try:
                        stats_list = p.get("stats")
                        # prefer the entry labelled 'Avg'
                        avg_entry = next((s for s in stats_list if str(s.get("statName", "")).lower() == "avg"), None)
                        if avg_entry:
                            stat_val = avg_entry.get("statValue") or avg_entry.get("value")
                        else:
                            stat_val = stats_list[0].get("statValue") or stats_list[0].get("value")
                    except Exception:
                        stat_val = stat_val

                rows.append({
                    "player_id":    player_id,
                    "name":         player_name,
                    "rank":         p.get("rank") or p.get("statRank"),
                    stat_name:      stat_val,
                    "events":       p.get("events") or p.get("eventCount"),
                    "year":         year,
                })
        except Exception:
            pass

    # --- Attempt 2: PGA Tour GraphQL endpoint ---
    if not rows:
        try:
            gql_url = "https://orchestrator.pgatour.com/graphql"
            payload = {
                "operationName": "StatDetails",
                "variables": {
                    "tourCode": "R",
                    "statId": stat_id,
                    "year": str(year),
                    "period": None,
                    "providerId": None,
                },
                "query": """
                    query StatDetails($tourCode: String!, $statId: String!, $year: String) {
                      statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
                        rows {
                          ... on StatDetailsPlayer {
                            plrId
                            plrName
                            rank
                            statValue
                            events
                          }
                        }
                      }
                    }
                """,
            }
            gql_resp = requests.post(
                gql_url,
                json=payload,
                headers={
                    "x-api-key": "da2-gsrx5bibzbb4njvhl7t37wqyl4",  # public read-only key
                    "content-type": "application/json",
                },
                timeout=15,
            )
            if gql_resp.status_code == 200:
                gql_data = gql_resp.json()
                player_rows = (
                    gql_data.get("data", {})
                    .get("statDetails", {})
                    .get("rows", [])
                )
                for p in player_rows:
                    rows.append({
                        "player_id": p.get("plrId"),
                        "name":      p.get("plrName"),
                        "rank":      p.get("rank"),
                        stat_name:   p.get("statValue"),
                        "events":    p.get("events"),
                        "year":      year,
                    })
        except Exception:
            pass

    # --- Attempt 3: HTML table fallback ---
    if not rows:
        print(f"  ⚠️  Falling back to HTML table parsing for {stat_name}/{year}")
        df_html = _parse_stat_table_html(resp.text, stat_name)
        if not df_html.empty:
            df_html["year"] = year
            df_html["stat_name_key"] = stat_name
            return df_html

    if not rows:
        print(f"  ⚠️  No data found for {stat_name}/{year}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Some stat values are strings with percent signs or commas (e.g. '86.11%').
    # Normalize those before coercion so percentages become numeric (86.11).
    if stat_name in df.columns:
        df[stat_name] = df[stat_name].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
    # Coerce stat, rank, events to numeric
    df[stat_name] = pd.to_numeric(df.get(stat_name), errors="coerce")
    df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")
    df["events"] = pd.to_numeric(df.get("events"), errors="coerce")

    # Clean player names
    df["name"] = df["name"].astype(str).str.strip()

    print(f"  ✓ {len(df)} players for {stat_name} ({year})")
    return df


def scrape_all_stats(year: int, stat_group: str = "all", use_cache: bool = True) -> pd.DataFrame:
    """
    Scrape all stats in a group for a given year and merge into one
    wide DataFrame (one row per player, one column per stat).

    Returns: DataFrame with player_id, name, year + one col per stat.
    """
    stat_names = STAT_GROUPS.get(stat_group, [stat_group])
    print(f"\n{'='*60}")
    print(f"Scraping PGA Tour stats – {year} season ({stat_group})")
    print(f"{'='*60}")

    merged = None

    for sname in stat_names:
        df = scrape_stat(sname, year, use_cache=use_cache)
        if df.empty:
            continue

        # Keep only the key columns for merging
        keep = ["player_id", "name", "year", sname]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].drop_duplicates(subset=["player_id"]) if "player_id" in df.columns else df[keep]

        if merged is None:
            merged = df
        else:
            merge_on = [c for c in ["player_id", "name", "year"] if c in df.columns and c in merged.columns]
            merged = merged.merge(df[merge_on + [sname]], on=merge_on, how="outer")

        time.sleep(0.3)  # polite delay between stats

    if merged is None:
        print(f"\n⚠️  No data scraped for {year}")
        return pd.DataFrame()

    print(f"\n✓ Stats scraped: {[c for c in merged.columns if c in STAT_IDS]}")
    print(f"  Players: {len(merged)}")
    return merged


def scrape_and_save(start_year: int, end_year: int,
                    stat_group: str = "all", use_cache: bool = True) -> None:
    """Scrape stats for a range of years and save parquet per year."""
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_years = []

    for year in range(start_year, end_year + 1):
        df = scrape_all_stats(year, stat_group=stat_group, use_cache=use_cache)
        if df.empty:
            continue

        out_path = out_dir / f"pga_stats_{year}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\n💾 Saved: {out_path} ({len(df)} rows)")
        all_years.append(df)

    if all_years:
        combined = pd.concat(all_years, ignore_index=True)
        combined_path = out_dir / "pga_stats_all.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"\n💾 Combined saved: {combined_path} ({len(combined)} rows)")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape PGA Tour player statistics")
    parser.add_argument("--year",  type=int, help="Single season year (e.g., 2024)")
    parser.add_argument("--start", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end",   type=int, default=2025, help="End year (default: 2025)")
    parser.add_argument("--stat",  type=str, default="all",
                        help=f"Stat group or single stat. Groups: {list(STAT_GROUPS)}. "
                             f"Single stats: {list(STAT_IDS)}")
    parser.add_argument("--no-cache", action="store_true", help="Disable HTTP cache")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.year:
        df = scrape_all_stats(args.year, stat_group=args.stat, use_cache=use_cache)
        if not df.empty:
            out = DATA_DIR / f"pga_stats_{args.year}.parquet"
            df.to_parquet(out, index=False)
            print(f"\n💾 Saved: {out}")
    else:
        scrape_and_save(args.start, args.end, stat_group=args.stat, use_cache=use_cache)

    print("\n✅ Done!")
