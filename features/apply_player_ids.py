"""
Apply player IDs to scraped tournament data.

This script:
1. Builds/updates the player registry from all scraped data
2. Applies player_id to each data file
3. Validates the results
"""

import pandas as pd
from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.player_ids import PlayerRegistry, build_registry_from_scraped_data


DATA_DIR = Path("data_files")


def apply_ids_to_file(file_path: Path, registry: PlayerRegistry, output_path: Path = None) -> pd.DataFrame:
    """
    Apply player IDs to a tournament data file.
    
    Args:
        file_path: Path to parquet file
        registry: PlayerRegistry instance
        output_path: Where to save updated file (defaults to overwriting input)
    
    Returns:
        Updated DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} rows")
    
    # Extract source name from filename (e.g., "espn_pga_2022.parquet" -> "espn")
    source = file_path.stem.split('_')[0]
    
    # Apply player IDs
    print("Applying player IDs...")
    df['player_id'] = df.apply(
        lambda row: registry.get_id(row['name']) or registry.add_player(
            name=row['name'],
            country=row.get('country'),
            source=source,
            check_duplicates=False
        ),
        axis=1
    )
    
    # Validation
    null_count = df['player_id'].isna().sum()
    unknown_count = (df['player_id'] == 'unknown').sum()
    unique_players = df['player_id'].nunique()
    
    print(f"\nResults:")
    print(f"  Unique players: {unique_players:,}")
    print(f"  Null IDs: {null_count:,}")
    print(f"  Unknown IDs: {unknown_count:,}")
    print(f"  Valid IDs: {len(df) - null_count - unknown_count:,}")
    
    # Show sample
    print(f"\nSample player IDs:")
    sample = df[['player_id', 'name', 'country']].drop_duplicates().head(10)
    for _, row in sample.iterrows():
        print(f"  {row['player_id']:12s} | {row['name']:25s} | {row.get('country', 'N/A')}")
    
    # Save
    if output_path is None:
        output_path = file_path
    
    df.to_parquet(output_path, index=False)
    print(f"\n💾 Saved to: {output_path.name}")
    
    return df


def process_all_espn_data():
    """Process all ESPN scraped data files."""
    # Find all ESPN data files
    espn_files = list(DATA_DIR.glob("espn_pga_*.parquet"))
    
    if not espn_files:
        print("No ESPN data files found in data_files/")
        return
    
    print(f"\nFound {len(espn_files)} ESPN data file(s):")
    for f in espn_files:
        print(f"  - {f.name}")
    
    # Build registry from all files
    print("\n" + "="*60)
    print("STEP 1: Building player registry")
    print("="*60)
    registry = build_registry_from_scraped_data(espn_files, source_name="espn")
    
    # Apply IDs to each file
    print("\n" + "="*60)
    print("STEP 2: Applying player IDs to data files")
    print("="*60)
    
    for file_path in espn_files:
        df = apply_ids_to_file(file_path, registry)
    
    # Final registry stats
    registry.stats()
    
    print("\n✅ Player ID mapping complete!")
    print(f"\nPlayer registry saved to: {registry.registry_path}")
    print(f"Updated data files: {len(espn_files)}")


def validate_player_ids():
    """Validate player IDs across all data files."""
    print("\n" + "="*60)
    print("Validating Player IDs")
    print("="*60 + "\n")
    
    # Load registry
    registry = PlayerRegistry()
    
    # Check all ESPN files
    espn_files = list(DATA_DIR.glob("espn_pga_*.parquet"))
    
    if not espn_files:
        print("No ESPN data files found")
        return
    
    all_issues = []
    
    for file_path in espn_files:
        print(f"Checking: {file_path.name}")
        df = pd.read_parquet(file_path)
        
        # Check for issues
        if 'player_id' not in df.columns:
            print(f"  ❌ Missing player_id column!")
            all_issues.append(f"{file_path.name}: missing player_id column")
            continue
        
        null_count = df['player_id'].isna().sum()
        unknown_count = (df['player_id'] == 'unknown').sum()
        
        if null_count > 0:
            print(f"  ⚠️  {null_count:,} null player IDs")
            all_issues.append(f"{file_path.name}: {null_count} null IDs")
        
        if unknown_count > 0:
            print(f"  ⚠️  {unknown_count:,} unknown player IDs")
            all_issues.append(f"{file_path.name}: {unknown_count} unknown IDs")
        
        # Check that all player_ids exist in registry
        unique_ids = df['player_id'].dropna().unique()
        missing_from_registry = []
        
        for pid in unique_ids:
            if pid != 'unknown':
                info = registry.get_player_info(pid)
                if info is None:
                    missing_from_registry.append(pid)
        
        if missing_from_registry:
            print(f"  ❌ {len(missing_from_registry)} IDs not in registry!")
            all_issues.append(f"{file_path.name}: {len(missing_from_registry)} orphaned IDs")
        
        if null_count == 0 and unknown_count == 0 and not missing_from_registry:
            print(f"  ✅ All {len(df):,} rows have valid player IDs")
    
    print("\n" + "="*60)
    if all_issues:
        print("Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("✅ All player IDs are valid!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply player IDs to tournament data")
    parser.add_argument('--validate', action='store_true', help="Only validate existing IDs")
    parser.add_argument('--file', type=str, help="Process a specific file")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_player_ids()
    elif args.file:
        registry = PlayerRegistry()
        file_path = Path(args.file)
        apply_ids_to_file(file_path, registry)
    else:
        process_all_espn_data()
