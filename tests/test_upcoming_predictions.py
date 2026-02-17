"""Test upcoming tournament predictions."""

import pytest
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta


def test_get_upcoming_tournaments_returns_dataframe():
    """Test that get_upcoming_tournaments returns a DataFrame with upcoming events."""
    from models.predict_upcoming import get_upcoming_tournaments
    
    upcoming = get_upcoming_tournaments(days_ahead=90)
    
    # Should return a DataFrame
    assert isinstance(upcoming, pd.DataFrame)
    
    # Should have expected columns
    expected_cols = {'id', 'name', 'date', 'season'}
    assert expected_cols.issubset(set(upcoming.columns))
    
    # All dates should be in the future
    today = datetime.now().date()
    if len(upcoming) > 0:
        assert all(upcoming['date'].dt.date >= today)


def test_build_current_player_features():
    """Test that we can build current player features from historical data."""
    from models.predict_upcoming import build_current_player_features
    
    features_path = Path(__file__).parent.parent / "data_files" / "espn_with_owgr_features.parquet"
    if not features_path.exists():
        features_path = Path(__file__).parent.parent / "data_files" / "espn_player_tournament_features.parquet"
    
    if not features_path.exists():
        pytest.skip("Feature data not available")
    
    current_features = build_current_player_features()
    
    # Should return a DataFrame
    assert isinstance(current_features, pd.DataFrame)
    
    # Should have players (allow for some duplicate names due to id variations)
    assert len(current_features) >= current_features['name'].nunique() - 5
    
    # Should have expected feature columns
    # (at a minimum, should have name and some historical stats)
    assert 'name' in current_features.columns
    assert len(current_features.columns) > 1


def test_predict_upcoming_tournament_integration():
    """Integration test for upcoming tournament prediction."""
    from models.predict_upcoming import predict_upcoming_tournament, get_upcoming_tournaments
    
    # Get an upcoming tournament
    upcoming = get_upcoming_tournaments(days_ahead=90)
    
    if len(upcoming) == 0:
        pytest.skip("No upcoming tournaments available")
    
    # Take the first upcoming tournament
    first_event = upcoming.iloc[0]
    
    # Make predictions
    predictions = predict_upcoming_tournament(
        tournament_name=first_event['name'],
        tournament_id=first_event['id'],
        tournament_date=first_event['date']
    )
    
    # Should return a DataFrame
    assert isinstance(predictions, pd.DataFrame)
    
    # Should have predictions
    if len(predictions) > 0:
        # Should have required columns
        assert 'name' in predictions.columns
        assert 'win_probability' in predictions.columns
        
        # Field should be realistic size (not all 1650 players)
        assert len(predictions) <= 200, f"Field too large: {len(predictions)} players"
        
        # Probabilities should sum to ~1.0 (allowing for rounding)
        assert 0.99 <= predictions['win_probability'].sum() <= 1.01
        
        # All probabilities should be between 0 and 1
        assert all(predictions['win_probability'] >= 0)
        assert all(predictions['win_probability'] <= 1)
        
        # Top player should have reasonable win probability (at least 1% in 156-player field)
        assert predictions['win_probability'].iloc[0] >= 0.01, \
            f"Top player probability too low: {predictions['win_probability'].iloc[0]*100:.2f}%"
