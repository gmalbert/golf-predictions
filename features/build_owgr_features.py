"""
Build OWGR Features for Tournament Predictions

For each player/tournament, adds OWGR features:
- Current rank at tournament time
- Rank momentum (4/12/52 week trends)
- Points trends
- Rank volatility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


def find_owgr_rank_at_date(player_id, tournament_date, owgr_df, weeks_back=0):
    """
    Find a player's OWGR rank at a specific date (or N weeks before).
    
    Args:
        player_id: Player ID
        tournament_date: Tournament date (datetime)
        owgr_df: OWGR rankings DataFrame
        weeks_back: How many weeks before the tournament date to look
        
    Returns:
        dict with rank and points data, or None if not found
    """
    # Calculate target date (remove timezone for comparison)
    target_date = pd.Timestamp(tournament_date).tz_localize(None) - timedelta(weeks=weeks_back)
    
    # If caller passed a pre-filtered per-player DataFrame or None, handle it quickly
    if owgr_df is None or len(owgr_df) == 0:
        return None

    if 'player_id' not in owgr_df.columns or (owgr_df['player_id'].nunique() == 1 and owgr_df['player_id'].dropna().iloc[0] == player_id):
        player_owgr = owgr_df.copy()
    else:
        # Fall back to filtering when a full OWGR DataFrame is provided
        player_owgr = owgr_df[owgr_df['player_id'] == player_id].copy()

    if player_owgr.empty:
        return None
    
    # Need to create a date column from year/week
    # OWGR weeks are calendar weeks, so we can approximate
    # Week 1 of each year starts around Jan 1
    player_owgr['approx_date'] = pd.to_datetime(
        player_owgr['source_year'].astype(str) + '-W' + player_owgr['source_week'].astype(str).str.zfill(2) + '-0',
        format='%Y-W%U-%w',
        errors='coerce'
    )
    
    # Drop rows where date couldn't be parsed
    player_owgr = player_owgr.dropna(subset=['approx_date'])
    
    if player_owgr.empty:
        return None
    
    # Find the ranking closest to (but not after) the target date
    player_owgr['days_diff'] = (player_owgr['approx_date'] - target_date).dt.days
    
    # Only consider rankings from before/at the tournament (not future rankings)
    valid_rankings = player_owgr[player_owgr['days_diff'] <= 0]
    
    if valid_rankings.empty:
        return None
    
    # Get the most recent ranking before the tournament
    closest = valid_rankings.loc[valid_rankings['days_diff'].idxmax()]
    
    return {
        'rank_this_week': closest['rank_this_week'],
        'avg_points': closest['avg_points'],
        'rank_last_week': closest['rank_last_week'],
        'total_points': closest['total_points'],
        'events_played_actual': closest['events_played_actual'],
        'owgr_data_date': closest['approx_date'],
        'days_before_tournament': -closest['days_diff']
    }


def build_owgr_features(espn_file='data_files/espn_player_tournament_features.parquet',
                       owgr_file='data_files/owgr_rankings_with_ids.parquet',
                       output_file='data_files/espn_with_owgr_features.parquet'):
    """
    Add OWGR features to ESPN tournament data.
    """
    print("\n" + "="*70)
    print("BUILDING OWGR FEATURES")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading ESPN data from {espn_file}")
    espn = pd.read_parquet(espn_file)
    print(f"   {len(espn):,} tournament records")
    
    print(f"\n📂 Loading OWGR data from {owgr_file}")
    owgr = pd.read_parquet(owgr_file)
    print(f"   {len(owgr):,} OWGR records")

    # Build a per-player index for fast lookups (much faster than filtering the full OWGR df each loop)
    print("\n🔧 Indexing OWGR by player_id for fast lookup...")
    owgr_map = {pid: grp.reset_index(drop=True) for pid, grp in owgr.groupby('player_id')}
    print(f"   Indexed {len(owgr_map):,} unique OWGR players")
    
    # Initialize new columns
    print(f"\n⚙️  Building OWGR features...")
    
    owgr_features = {
        'owgr_rank_current': [],
        'owgr_rank_4w_ago': [],
        'owgr_rank_12w_ago': [],
        'owgr_rank_52w_ago': [],
        'owgr_points_current': [],
        'owgr_rank_change_4w': [],
        'owgr_rank_change_12w': [],
        'owgr_rank_change_52w': [],
        'owgr_data_staleness_days': [],
    }
    
    # Process each tournament record
    total = len(espn)
    found_count = {0: 0, 4: 0, 12: 0, 52: 0}
    
    for idx, row in espn.iterrows():
        player_id = row['player_id']
        tournament_date = row['date']
        
        # Get OWGR ranks at different time points
        # Use pre-indexed per-player OWGR frames for fast lookup
        player_owgr_df = owgr_map.get(player_id)
        current = find_owgr_rank_at_date(player_id, tournament_date, player_owgr_df, weeks_back=0)
        rank_4w = find_owgr_rank_at_date(player_id, tournament_date, player_owgr_df, weeks_back=4)
        rank_12w = find_owgr_rank_at_date(player_id, tournament_date, player_owgr_df, weeks_back=12)
        rank_52w = find_owgr_rank_at_date(player_id, tournament_date, player_owgr_df, weeks_back=52)
        
        # Current rank and points
        if current:
            owgr_features['owgr_rank_current'].append(current['rank_this_week'])
            owgr_features['owgr_points_current'].append(current['avg_points'])
            owgr_features['owgr_data_staleness_days'].append(current['days_before_tournament'])
            found_count[0] += 1
        else:
            owgr_features['owgr_rank_current'].append(np.nan)
            owgr_features['owgr_points_current'].append(np.nan)
            owgr_features['owgr_data_staleness_days'].append(np.nan)
        
        # Historical ranks
        if rank_4w:
            owgr_features['owgr_rank_4w_ago'].append(rank_4w['rank_this_week'])
            found_count[4] += 1
        else:
            owgr_features['owgr_rank_4w_ago'].append(np.nan)
            
        if rank_12w:
            owgr_features['owgr_rank_12w_ago'].append(rank_12w['rank_this_week'])
            found_count[12] += 1
        else:
            owgr_features['owgr_rank_12w_ago'].append(np.nan)
            
        if rank_52w:
            owgr_features['owgr_rank_52w_ago'].append(rank_52w['rank_this_week'])
            found_count[52] += 1
        else:
            owgr_features['owgr_rank_52w_ago'].append(np.nan)
        
        # Rank changes (momentum indicators)
        # Negative = improved (rank went down in number)
        if current and rank_4w:
            owgr_features['owgr_rank_change_4w'].append(
                current['rank_this_week'] - rank_4w['rank_this_week']
            )
        else:
            owgr_features['owgr_rank_change_4w'].append(np.nan)
            
        if current and rank_12w:
            owgr_features['owgr_rank_change_12w'].append(
                current['rank_this_week'] - rank_12w['rank_this_week']
            )
        else:
            owgr_features['owgr_rank_change_12w'].append(np.nan)
            
        if current and rank_52w:
            owgr_features['owgr_rank_change_52w'].append(
                current['rank_this_week'] - rank_52w['rank_this_week']
            )
        else:
            owgr_features['owgr_rank_change_52w'].append(np.nan)
        
        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"   Processed {idx + 1:,}/{total:,} records...")
    
    # Add features to DataFrame
    for feature_name, values in owgr_features.items():
        espn[feature_name] = values
    
    # Print statistics
    print(f"\n📊 Feature Coverage:")
    print(f"   Current rank:  {found_count[0]:,}/{total:,} ({found_count[0]/total*100:.1f}%)")
    print(f"   4-week rank:   {found_count[4]:,}/{total:,} ({found_count[4]/total*100:.1f}%)")
    print(f"   12-week rank:  {found_count[12]:,}/{total:,} ({found_count[12]/total*100:.1f}%)")
    print(f"   52-week rank:  {found_count[52]:,}/{total:,} ({found_count[52]/total*100:.1f}%)")
    
    # Show sample
    print(f"\n📋 Sample Data:")
    sample_cols = ['name', 'tournament', 'date', 'owgr_rank_current', 
                   'owgr_rank_change_4w', 'owgr_rank_change_52w', 'tournament_rank']
    print(espn[sample_cols].head(10).to_string())
    
    # Save
    print(f"\n💾 Saving enhanced data to {output_file}...")
    espn.to_parquet(output_file, index=False)
    
    file_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"   File size: {file_size:.2f} MB")
    
    print(f"\n✅ SUCCESS!")
    print("="*70)
    
    return espn


def analyze_owgr_impact():
    """Quick analysis of OWGR feature distributions."""
    print("\n" + "="*70)
    print("OWGR FEATURE ANALYSIS")
    print("="*70)
    
    df = pd.read_parquet('data_files/espn_with_owgr_features.parquet')
    
    # Filter to records with OWGR data
    with_owgr = df[df['owgr_rank_current'].notna()]
    
    print(f"\n📊 Records with OWGR data: {len(with_owgr):,}/{len(df):,} ({len(with_owgr)/len(df)*100:.1f}%)")
    
    print(f"\n📈 OWGR Rank Distribution:")
    print(f"   Mean rank: {with_owgr['owgr_rank_current'].mean():.1f}")
    print(f"   Median rank: {with_owgr['owgr_rank_current'].median():.1f}")
    print(f"   Rank range: {with_owgr['owgr_rank_current'].min():.0f} to {with_owgr['owgr_rank_current'].max():.0f}")
    
    print(f"\n📉 Rank Momentum (4-week change):")
    momentum = with_owgr['owgr_rank_change_4w'].dropna()
    print(f"   Mean change: {momentum.mean():.1f} (negative = improving)")
    print(f"   Median change: {momentum.median():.1f}")
    print(f"   Improving (negative): {(momentum < 0).sum():,} ({(momentum < 0).sum()/len(momentum)*100:.1f}%)")
    print(f"   Declining (positive): {(momentum > 0).sum():,} ({(momentum > 0).sum()/len(momentum)*100:.1f}%)")
    
    print(f"\n🎯 OWGR vs Tournament Performance:")
    correlation = with_owgr[['owgr_rank_current', 'tournament_rank']].corr().iloc[0, 1]
    print(f"   Correlation (rank vs finish): {correlation:.3f}")
    print(f"   {'Strong positive correlation!' if correlation > 0.5 else 'Moderate correlation' if correlation > 0.3 else 'Weak correlation'}")
    
    # Top performers
    print(f"\n🏆 Tournament Winners - OWGR Ranks:")
    winners = with_owgr[with_owgr['tournament_rank'] == 1]
    print(f"   Winners with OWGR data: {len(winners)}")
    print(f"   Average winner OWGR rank: {winners['owgr_rank_current'].mean():.1f}")
    print(f"   Median winner OWGR rank: {winners['owgr_rank_current'].median():.1f}")
    print(f"   Winner rank range: {winners['owgr_rank_current'].min():.0f} to {winners['owgr_rank_current'].max():.0f}")
    
    print("="*70)


if __name__ == '__main__':
    # Build features
    df = build_owgr_features()
    
    # Analyze
    analyze_owgr_impact()
