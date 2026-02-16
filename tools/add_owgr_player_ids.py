"""
Add OWGR players to the player registry and link to existing ESPN player IDs.

This script:
1. Loads existing player registry
2. Adds all OWGR players to registry
3. Links OWGR players to ESPN player_ids where names match
4. Adds player_id column to OWGR data
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.player_ids import PlayerRegistry, normalize_name, name_similarity


def analyze_overlap():
    """Analyze name overlap between ESPN and OWGR data."""
    print("\n" + "="*70)
    print("ANALYZING ESPN vs OWGR NAME OVERLAP")
    print("="*70)
    
    espn = pd.read_parquet('data_files/espn_pga_2018_2025.parquet')
    owgr = pd.read_parquet('data_files/owgr_rankings.parquet')
    
    print(f"\n📊 ESPN Data:")
    print(f"   Total rows: {len(espn):,}")
    print(f"   Unique players: {espn['name'].nunique()}")
    
    print(f"\n📊 OWGR Data:")
    print(f"   Total rows: {len(owgr):,}")
    print(f"   Unique players: {owgr['player_name'].nunique():,}")
    
    # Check overlap
    espn_names_norm = set(espn['name'].apply(normalize_name).unique())
    owgr_names_norm = set(owgr['player_name'].apply(normalize_name).unique())
    
    overlap = espn_names_norm & owgr_names_norm
    
    print(f"\n🔗 Name Overlap (normalized):")
    print(f"   Exact matches: {len(overlap)} players")
    print(f"   ESPN-only: {len(espn_names_norm - owgr_names_norm)}")
    print(f"   OWGR-only: {len(owgr_names_norm - espn_names_norm):,}")
    
    # Show some overlapping names
    print(f"\n   Sample overlapping players:")
    for name in sorted(overlap)[:10]:
        print(f"      - {name}")
    
    return overlap


def add_owgr_to_registry():
    """Add all OWGR players to the player registry."""
    print("\n" + "="*70)
    print("ADDING OWGR PLAYERS TO REGISTRY")
    print("="*70)
    
    # Load data
    owgr = pd.read_parquet('data_files/owgr_rankings.parquet')
    
    # Get unique players from OWGR
    unique_players = owgr[['player_name', 'country']].drop_duplicates()
    print(f"\n📝 Found {len(unique_players):,} unique OWGR players")
    
    # Load registry
    registry = PlayerRegistry()
    initial_count = len(registry.players)
    
    # Add each player
    print(f"\n⚙️  Adding players to registry...")
    player_ids = {}
    
    for idx, row in unique_players.iterrows():
        name = row['player_name']
        country = row['country']
        
        # Add to registry (will return existing ID if already present)
        player_id = registry.add_player(
            name=name,
            country=country,
            source='owgr',
            check_duplicates=False  # Skip warnings for speed
        )
        
        player_ids[name] = player_id
        
        if (idx + 1) % 1000 == 0:
            print(f"   Processed {idx + 1:,}/{len(unique_players):,} players...")
    
    # Save registry
    registry.save()
    
    added_count = len(registry.players) - initial_count
    print(f"\n✅ Added {added_count:,} new players to registry")
    print(f"   Total players in registry: {len(registry.players):,}")
    
    return player_ids


def link_espn_ids():
    """Link OWGR players to existing ESPN player IDs where names match."""
    print("\n" + "="*70)
    print("LINKING ESPN PLAYER IDS")
    print("="*70)
    
    # Load data
    espn = pd.read_parquet('data_files/espn_pga_2018_2025.parquet')
    registry_df = pd.read_parquet('data_files/player_registry.parquet')
    
    # Create mapping of normalized name -> ESPN player_id
    espn_unique = espn[['name', 'player_id']].drop_duplicates()
    espn_unique['normalized_name'] = espn_unique['name'].apply(normalize_name)
    espn_id_map = dict(zip(espn_unique['normalized_name'], espn_unique['player_id']))
    
    print(f"\n📋 ESPN player ID mapping has {len(espn_id_map)} players")
    
    # Update registry with ESPN IDs where they match
    registry_df['espn_player_id'] = registry_df['normalized_name'].map(espn_id_map)
    
    matches = registry_df['espn_player_id'].notna().sum()
    print(f"✅ Matched {matches:,} OWGR players to ESPN IDs")
    
    # Save updated registry
    registry_df.to_parquet('data_files/player_registry.parquet', index=False)
    print(f"💾 Updated registry saved")
    
    return registry_df


def add_player_ids_to_owgr():
    """Add player_id column to OWGR data."""
    print("\n" + "="*70)
    print("ADDING PLAYER IDS TO OWGR DATA")
    print("="*70)
    
    # Load data
    owgr = pd.read_parquet('data_files/owgr_rankings.parquet')
    registry_df = pd.read_parquet('data_files/player_registry.parquet')
    
    # Create mapping: player_name -> player_id
    # First normalize OWGR names
    owgr['normalized_name'] = owgr['player_name'].apply(normalize_name)
    
    # Create mapping from registry
    id_map = dict(zip(registry_df['normalized_name'], registry_df['player_id']))
    
    # Map player IDs
    owgr['player_id'] = owgr['normalized_name'].map(id_map)
    
    # Drop temporary column
    owgr = owgr.drop(columns=['normalized_name'])
    
    # Check results
    matched = owgr['player_id'].notna().sum()
    print(f"\n✅ Assigned player_id to {matched:,}/{len(owgr):,} rows ({matched/len(owgr)*100:.2f}%)")
    
    # Save updated OWGR data
    output_file = 'data_files/owgr_rankings_with_ids.parquet'
    owgr.to_parquet(output_file, index=False)
    
    file_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"💾 Saved to {output_file}")
    print(f"   File size: {file_size:.2f} MB")
    
    # Show sample
    print(f"\n📋 Sample data:")
    print(owgr[['player_id', 'player_name', 'rank_this_week', 'avg_points', 'source_year']].head(10))
    
    return owgr


def main():
    """Run the full player ID linkage process."""
    print("\n" + "="*70)
    print("🎯 OWGR PLAYER ID LINKAGE")
    print("="*70)
    
    # Step 1: Analyze overlap
    analyze_overlap()
    
    # Step 2: Add OWGR players to registry
    add_owgr_to_registry()
    
    # Step 3: Link ESPN IDs
    link_espn_ids()
    
    # Step 4: Add player IDs to OWGR data
    add_player_ids_to_owgr()
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use owgr_rankings_with_ids.parquet for OWGR data")
    print("  2. Join on player_id to link ESPN and OWGR data")
    print("  3. Build OWGR features (rank, momentum, etc.)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
