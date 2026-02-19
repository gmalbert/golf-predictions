"""
Upcoming Tournament Predictions

Fetch upcoming PGA Tour schedule and generate predictions using current player features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scrapers.espn_golf import get_espn_schedule


DATA_DIR = Path(__file__).parent.parent / 'data_files'


def get_upcoming_tournaments(days_ahead=90):
    """
    Fetch upcoming PGA tournaments from ESPN.
    
    Args:
        days_ahead: How many days ahead to look (default 90)
    
    Returns:
        DataFrame with upcoming tournaments
    """
    current_year = datetime.now().year
    next_year = current_year + 1
    
    # Get this year and next year's schedule
    events = []
    for year in [current_year, next_year]:
        try:
            year_events = get_espn_schedule(year)
            events.extend(year_events)
        except Exception as e:
            print(f"Warning: Could not fetch {year} schedule: {e}")
    
    if not events:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    # Filter to upcoming tournaments only
    now = pd.Timestamp.now(tz='UTC')
    cutoff = now + timedelta(days=days_ahead)
    
    upcoming = df[(df['date'] >= now) & (df['date'] <= cutoff)].copy()
    upcoming = upcoming.sort_values('date').reset_index(drop=True)
    
    print(f"Found {len(upcoming)} upcoming tournaments in next {days_ahead} days")
    return upcoming


def build_current_player_features(historical_features_file=None):
    """
    Build player features using most recent historical data.
    
    For each player, uses their latest tournament features as baseline
    for predicting future tournaments.
    
    Args:
        historical_features_file: Path to historical features parquet
    
    Returns:
        DataFrame with current player features
    """
    if historical_features_file is None:
        ext_path = DATA_DIR / 'espn_with_extended_features.parquet'
        owgr_path = DATA_DIR / 'espn_with_owgr_features.parquet'
        base_path = DATA_DIR / 'espn_player_tournament_features.parquet'
        if ext_path.exists():
            historical_features_file = ext_path
        elif owgr_path.exists():
            historical_features_file = owgr_path
        else:
            historical_features_file = base_path
    
    if not historical_features_file.exists():
        raise FileNotFoundError(f"Historical features not found: {historical_features_file}")
    
    print(f"Loading historical features from: {historical_features_file}")
    df = pd.read_parquet(historical_features_file)
    
    # Get most recent record for each player
    df['date'] = pd.to_datetime(df['date'], utc=True)
    # Drop rows with NaT dates (placeholder sentinel rows) so tail(1) picks the
    # actual most-recent tournament record, not a dateless 2018 stub.
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    
    # Take latest record per player
    latest = df.groupby('player_id').tail(1).copy()

    # ── Staleness penalty ────────────────────────────────────────────────────
    # If a player's most-recent record is >90 days old, regress short-window
    # form features toward the field median so stale data doesn't over-rank
    # players whose quality we can no longer accurately measure.
    now = pd.Timestamp.now(tz='UTC')
    latest['_data_age_days'] = (now - latest['date']).dt.days

    FORM_COLS = [
        'prior_avg_score_5', 'prior_avg_score_10',
        'prior_std_score_5', 'prior_std_score_10',
        'prior_top10_rate_5', 'prior_top10_rate_10',
        'last_event_score', 'last_event_rank',
        'season_to_date_avg_score',
        'sg_total_season', 'sg_off_tee_season', 'sg_approach_season',
        'sg_around_green_season', 'sg_putting_season',
        'driving_distance_season', 'gir_pct_season',
    ]
    existing_form = [c for c in FORM_COLS if c in latest.columns]

    if existing_form:
        field_medians = latest[existing_form].median()
        STALE_THRESHOLD = 90   # days
        MAX_AGE         = 365  # days – full regression at 1 year

        def staleness_alpha(age_days):
            """Linear decay: 0 at 90d -> 1.0 at 365d (full regression to median)."""
            if age_days <= STALE_THRESHOLD:
                return 0.0
            return min(1.0, (age_days - STALE_THRESHOLD) / (MAX_AGE - STALE_THRESHOLD))

        alphas = latest['_data_age_days'].apply(staleness_alpha)
        stale_mask = alphas > 0
        n_stale = stale_mask.sum()
        if n_stale > 0:
            print(f"[INFO] Applying staleness regression to {n_stale} players "
                  f"(>{STALE_THRESHOLD}d since last record)")
            for col in existing_form:
                med = field_medians[col]
                latest.loc[stale_mask, col] = (
                    latest.loc[stale_mask, col] * (1 - alphas[stale_mask])
                    + med * alphas[stale_mask]
                )

    latest = latest.drop(columns=['_data_age_days'])

    print(f"Built current features for {len(latest)} players")
    return latest


def predict_upcoming_tournament(tournament_name, tournament_id=None, tournament_date=None, field_size=156):
    """
    Generate predictions for an upcoming tournament.
    
    Uses current player features (from most recent tournaments) to predict
    who will win an upcoming event. Filters to a realistic field size.
    
    Args:
        tournament_name: Name of upcoming tournament
        tournament_id: ESPN tournament ID (optional)
        tournament_date: Tournament date (optional)
        field_size: Expected field size (default 156, typical PGA Tour full-field event)
    
    Returns:
        DataFrame with predictions
    """
    from models.predict_tournament import predict_field, load_model
    
    print(f"\n{'='*70}")
    print(f"PREDICTING: {tournament_name} (UPCOMING)")
    print(f"{'='*70}\n")
    
    # Load model
    model, feature_cols = load_model()
    
    # Build current player features
    current_features = build_current_player_features()
    
    # Filter to realistic tournament field
    # Strategy: Use OWGR rank if available, otherwise use recent performance
    print(f"[INFO] Filtering {len(current_features)} players to tournament field of ~{field_size}")
    
    if 'owgr_rank_current' in current_features.columns:
        # Prefer players with OWGR ranking (active tour players)
        has_owgr = current_features[current_features['owgr_rank_current'].notna()].copy()
        no_owgr = current_features[current_features['owgr_rank_current'].isna()].copy()
        
        # Take top players by OWGR, fill remaining slots with recent performers
        field_from_owgr = min(field_size, len(has_owgr))
        remaining_slots = field_size - field_from_owgr
        
        has_owgr = has_owgr.nsmallest(field_from_owgr, 'owgr_rank_current')
        
        if remaining_slots > 0 and len(no_owgr) > 0:
            # Fill remaining slots with players sorted by recent performance
            # Use rolling average finish or tournament count as proxy
            if 'avg_finish_last_5' in no_owgr.columns:
                no_owgr = no_owgr.nsmallest(min(remaining_slots, len(no_owgr)), 'avg_finish_last_5')
            else:
                no_owgr = no_owgr.head(remaining_slots)
            
            current_features = pd.concat([has_owgr, no_owgr], ignore_index=True)
        else:
            current_features = has_owgr
    else:
        # No OWGR data - filter by recent activity and performance
        if 'avg_finish_last_5' in current_features.columns:
            current_features = current_features.nsmallest(field_size, 'avg_finish_last_5')
        else:
            # Fallback: take players with most recent tournaments
            current_features = current_features.head(field_size)
    
    print(f"[OK] Tournament field: {len(current_features)} players")
    
    # Add tournament context and calculate tournament-specific features
    current_features['tournament'] = tournament_name
    if tournament_date:
        current_features['date'] = pd.to_datetime(tournament_date, utc=True)
    else:
        current_features['date'] = pd.Timestamp.now(tz='UTC')
    
    # Calculate course history for this specific tournament
    print(f"[INFO] Calculating course history for {tournament_name}")
    historical_df = pd.read_parquet(DATA_DIR / 'espn_with_owgr_features.parquet' 
                                   if (DATA_DIR / 'espn_with_owgr_features.parquet').exists() 
                                   else DATA_DIR / 'espn_player_tournament_features.parquet')
    
    # Find historical performances at this tournament
    # Match by tournament name (case-insensitive partial match)
    tournament_history = historical_df[
        historical_df['tournament'].str.contains(tournament_name, case=False, na=False, regex=False)
    ]
    
    if len(tournament_history) > 0:
        # Calculate average score at this tournament for each player
        player_course_history = tournament_history.groupby('player_id').agg({
            'numeric_total_score': 'mean',  # Average score at this tournament
            'tournament_rank': 'mean'  # Average finish at this tournament
        }).reset_index()
        player_course_history.columns = ['player_id', 'course_avg_score', 'course_avg_finish']
        
        # Merge into current features
        current_features = current_features.merge(player_course_history, on='player_id', how='left')
        
        # Update the course_history_avg_score feature
        if 'course_history_avg_score' in current_features.columns:
            current_features['course_history_avg_score'] = current_features['course_avg_score'].fillna(
                current_features['course_history_avg_score']
            )
        else:
            current_features['course_history_avg_score'] = current_features['course_avg_score']
        
        # Drop temporary columns
        current_features = current_features.drop(columns=['course_avg_score', 'course_avg_finish'], errors='ignore')
        
        players_with_history = current_features['course_history_avg_score'].notna().sum()
        print(f"[INFO] {players_with_history}/{len(current_features)} players have history at {tournament_name}")
    else:
        print(f"[INFO] No historical data found for {tournament_name} - using general features only")
    
    # Make predictions
    predictions = predict_field(current_features, model, feature_cols)
    
    # Apply OWGR-based adjustment to better differentiate elite players
    # The raw model predictions are too flat - boost elite players
    if 'owgr_points_current' in predictions.columns and 'owgr_rank_current' in predictions.columns:
        print("[INFO] Applying OWGR-based adjustment to predictions")
        
        # Use OWGR points as a quality multiplier
        # Players with higher OWGR points get boosted probabilities
        # Use 0.7 power to create meaningful separation between elite and mid-tier players
        predictions['quality_multiplier'] = predictions['owgr_points_current'].fillna(0.1) ** 0.7
        
        # For players without OWGR data, use a baseline multiplier
        predictions.loc[predictions['owgr_points_current'].isna(), 'quality_multiplier'] = 0.5
        
        # Apply multiplier to win probabilities
        predictions['win_probability'] = predictions['win_probability'] * predictions['quality_multiplier']
        
        # Drop the temporary column
        predictions = predictions.drop(columns=['quality_multiplier'])
    
    # Normalize probabilities to sum to 1.0
    total_prob = predictions['win_probability'].sum()
    if total_prob > 0:
        predictions['win_probability'] = predictions['win_probability'] / total_prob
    
    # Re-sort by updated probabilities
    predictions = predictions.sort_values('win_probability', ascending=False).reset_index(drop=True)
    
    return predictions


if __name__ == '__main__':
    # Show upcoming tournaments
    upcoming = get_upcoming_tournaments()
    
    if len(upcoming) > 0:
        print("\n" + "="*70)
        print("UPCOMING TOURNAMENTS")
        print("="*70)
        for _, row in upcoming.head(10).iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            print(f"  {date_str}  {row['name']}")
        
        # Predict first upcoming tournament
        if len(upcoming) > 0:
            first = upcoming.iloc[0]
            print(f"\n\nGenerating predictions for: {first['name']}")
            
            predictions = predict_upcoming_tournament(
                first['name'],
                first['id'],
                first['date']
            )
            
            print(f"\n>> Top 10 Predictions:")
            top10 = predictions[['name', 'win_probability']].head(10)
            top10['win_probability'] = top10['win_probability'].apply(lambda x: f"{x:.2%}")
            print(top10.to_string(index=False))
    else:
        print("No upcoming tournaments found")
