"""
Check player ID assignments and show statistics.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.player_ids import PlayerRegistry


DATA_DIR = Path("data_files")


def analyze_player_ids():
    """Analyze player ID assignments in the ESPN data."""
    
    # Load data
    espn_file = DATA_DIR / "espn_pga_2022.parquet"
    df = pd.read_parquet(espn_file)
    
    print("\n" + "="*70)
    print("ESPN 2022 Data - Player ID Analysis")
    print("="*70 + "\n")
    
    print(f"Total rows: {len(df):,}")
    print(f"Tournaments: {df['tournament'].nunique()}")
    print(f"Unique players: {df['player_id'].nunique()}")
    
    # Handle date field (may have mixed types)
    valid_dates = df['date'].dropna()
    if len(valid_dates) > 0:
        print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
    
    # Check player_id column
    print(f"\nPlayer ID field:")
    print(f"  Null values: {df['player_id'].isna().sum():,} ({df['player_id'].isna().mean()*100:.1f}%)")
    print(f"  Unknown IDs: {(df['player_id'] == 'unknown').sum():,}")
    print(f"  Valid IDs: {df['player_id'].notna().sum():,} ({df['player_id'].notna().mean()*100:.1f}%)")
    
    # Top players by appearances
    print(f"\nTop 10 players by tournament appearances:")
    player_appearances = df.groupby(['player_id', 'name']).size().reset_index(name='tournaments')
    player_appearances = player_appearances.sort_values('tournaments', ascending=False)
    
    for idx, (_, row) in enumerate(player_appearances.head(10).iterrows(), 1):
        print(f"  {idx:2d}. {row['name']:30s} - {row['tournaments']:2d} tournaments - ID: {row['player_id']}")
    
    # Player ID stability check
    print(f"\nPlayer ID stability check:")
    name_to_ids = df.groupby('name')['player_id'].nunique()
    multi_id_players = name_to_ids[name_to_ids > 1]
    
    if len(multi_id_players) > 0:
        print(f"  ⚠️  {len(multi_id_players)} players have multiple IDs!")
        for name in multi_id_players.head(5).index:
            ids = df[df['name'] == name]['player_id'].unique()
            print(f"    {name}: {ids}")
    else:
        print(f"  ✅ All players have exactly 1 unique ID (stable mapping)")
    
    # Country distribution
    print(f"\nTop 10 countries by player count:")
    player_countries = df[['player_id', 'country']].drop_duplicates()
    country_counts = player_countries['country'].value_counts().head(10)
    
    for country, count in country_counts.items():
        print(f"  {country:20s}: {count:2d} players")
    
    # Sample records
    print(f"\nSample records (first 5 from a random tournament):")
    sample_tournament = df['tournament'].iloc[100]  # Pick arbitrary tournament
    sample = df[df['tournament'] == sample_tournament].head(5)
    
    print(f"\nTournament: {sample_tournament}")
    for _, row in sample.iterrows():
        print(f"  {row['player_id']:12s} | {row['name']:25s} | Score: {row['total_score']:6s} | Country: {row['country']}")
    
    print("\n" + "="*70 + "\n")


def check_registry():
    """Check the player registry file."""
    
    registry = PlayerRegistry()
    
    print("\n" + "="*70)
    print("Player Registry")
    print("="*70 + "\n")
    
    print(f"Registry file: {registry.registry_path}")
    print(f"Total players: {len(registry.players):,}")
    
    print(f"\nSample players:")
    sample = registry.players.head(10)
    
    for _, row in sample.iterrows():
        print(f"  {row['player_id']:12s} | {row['name']:30s} | {row['normalized_name']:30s} | {row['country']}")
    
    print(f"\nCountry breakdown:")
    country_counts = registry.players['country'].value_counts()
    for country, count in country_counts.items():
        print(f"  {country:20s}: {count:3d}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    analyze_player_ids()
    check_registry()
