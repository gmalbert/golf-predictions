"""
Merge per-year ESPN parquet files into a single consolidated parquet (and optional partitioned dataset).

Usage:
    python features/merge_espn_parquets.py --out data_files/espn_pga_2018_2025.parquet --partition data_files/espn_pga_all

The script will:
- Read all `data_files/espn_pga_*.parquet`
- Concatenate, sort, dedupe (by tournament + name + date)
- Save one consolidated file and optionally a partitioned dataset by `year`
- Print summary statistics
"""

from pathlib import Path
import pandas as pd
import argparse

DATA_DIR = Path("data_files")


def merge_parquets(out_path: Path, partition_dir: Path | None = None, dedupe: bool = True) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("espn_pga_*.parquet"))
    if not files:
        raise FileNotFoundError("No espn_pga_*.parquet files found in data_files/")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['source_file'] = f.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Normalize columns: ensure year column exists
    if 'year' not in combined.columns:
        combined['year'] = pd.DatetimeIndex(combined['date']).year

    # Optional dedupe
    if dedupe:
        before = len(combined)
        combined = combined.drop_duplicates(subset=['tournament_id', 'name', 'date', 'player_id'], keep='first')
        after = len(combined)
        print(f"Dropped {before-after:,} duplicate rows")

    # Sort for readability
    if 'date' in combined.columns:
        combined = combined.sort_values(['date', 'tournament', 'player_id']).reset_index(drop=True)

    # Save single parquet
    combined.to_parquet(out_path, index=False)
    print(f"💾 Saved consolidated parquet: {out_path} ({len(combined):,} rows)")

    # Optionally save partitioned dataset
    if partition_dir:
        partition_dir = Path(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(partition_dir, partition_cols=['year'], index=False)
        print(f"💾 Saved partitioned dataset to: {partition_dir} (partitioned by year)")

    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge ESPN parquet files')
    parser.add_argument('--out', type=str, default=str(DATA_DIR / 'espn_pga_2018_2025.parquet'))
    parser.add_argument('--partition', type=str, help='Optional directory to write partitioned parquet (by year)')
    parser.add_argument('--no-dedupe', action='store_true', help='Disable deduplication')

    args = parser.parse_args()
    out_path = Path(args.out)
    part_dir = Path(args.partition) if args.partition else None

    merged = merge_parquets(out_path, partition_dir=part_dir, dedupe=not args.no_dedupe)

    # Print quick summary
    print('\nSummary:')
    # Convert numpy year types to native Python ints for nicer display
    years_list = sorted(merged['year'].astype(int).unique().tolist())
    print(f"  Years included: {years_list}")
    print(f"  Tournaments: {merged['tournament'].nunique():,}")
    print(f"  Rows: {len(merged):,}")
    print(f"  Unique players: {merged['player_id'].nunique():,}")
