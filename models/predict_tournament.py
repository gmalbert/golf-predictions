"""
Tournament Winner Predictions

Load trained model and make predictions for specific tournaments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import xgboost as xgb


DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODEL_DIR = Path(__file__).parent / 'saved_models'


def load_model():
    """Load the trained model and feature list."""
    model_path = MODEL_DIR / 'winner_predictor_v2.joblib'
    features_path = MODEL_DIR / 'model_features_v2.txt'
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run: python models/train_improved_model.py"
        )
    
    # Prefer XGBoost-native model if available (avoids pickle/version issues)
    xgb_native_candidates = [
        MODEL_DIR / 'winner_predictor_v2.json',
        MODEL_DIR / 'winner_predictor_v2.xgb',
        MODEL_DIR / 'winner_predictor_v2.model'
    ]

    for candidate in xgb_native_candidates:
        if candidate.exists():
            try:
                clf = xgb.XGBClassifier()
                clf.load_model(str(candidate))
                feature_cols = features_path.read_text().strip().split('\n')
                print(f"[OK] Loaded XGBoost-native model from {candidate}")
                print(f"[OK] Using {len(feature_cols)} features")
                return clf, feature_cols
            except Exception:
                # Fall back to joblib if native load fails
                pass

    # Fallback: joblib (keeps backward compatibility)
    model = joblib.load(model_path)
    feature_cols = features_path.read_text().strip().split('\n')

    print(f"[OK] Loaded model from {model_path}")
    print(f"[OK] Using {len(feature_cols)} features")

    return model, feature_cols


def predict_field(field_df, model=None, feature_cols=None):
    """
    Predict win probabilities for a tournament field.
    
    Args:
        field_df: DataFrame with player data (must have all required features)
        model: Trained model (optional - will load if not provided)
        feature_cols: List of feature column names (optional - will load if not provided)
    
    Returns:
        DataFrame with predictions and probabilities
    """
    # Load model if not provided
    if model is None or feature_cols is None:
        model, feature_cols = load_model()
    
    # Ensure all features are present - add missing ones with zeros
    missing_features = [f for f in feature_cols if f not in field_df.columns]
    if missing_features:
        print(f"[WARNING] Missing {len(missing_features)} features, filling with zeros: {missing_features[:3]}...")
        for feat in missing_features:
            field_df[feat] = 0.0
    
    # Make predictions
    X = field_df[feature_cols].fillna(0)
    
    # Get probabilities
    probabilities = model.predict_proba(X)
    win_probs = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
    
    # Create results DataFrame - preserve ALL original columns
    results = field_df.copy()
    results['win_probability'] = win_probs
    results['predicted_win'] = (win_probs >= 0.5).astype(int)
    
    # Sort by win probability
    results = results.sort_values('win_probability', ascending=False).reset_index(drop=True)
    
    return results


def predict_tournament(tournament_name, year=2024):
    """
    Predict winners for a specific tournament.
    
    Args:
        tournament_name: Name of the tournament
        year: Tournament year
    
    Returns:
        DataFrame with predictions sorted by win probability
    """
    print("\n" + "="*70)
    print(f"PREDICTING: {tournament_name}")
    print("="*70)
    
    # Load model
    model, feature_cols = load_model()
    
    # Load tournament data
    data_file = DATA_DIR / 'espn_with_owgr_features.parquet'
    df = pd.read_parquet(data_file)
    
    # Filter to specific tournament and year
    tournament_data = df[
        (df['tournament'] == tournament_name) & 
        (df['year'] == year)
    ].copy()
    
    if len(tournament_data) == 0:
        raise ValueError(f"No data found for {tournament_name} {year}")
    
    print(f"\n[OK] Found {len(tournament_data)} players in {tournament_name} {year}")
    
    # Make predictions
    predictions = predict_field(tournament_data, model, feature_cols)
    
    print(f"\n>> Top 10 Predicted Winners:")
    
    # Build display columns dynamically based on what's available
    display_cols = ['name', 'win_probability']
    col_renames = {'name': 'Player', 'win_probability': 'Win Probability'}
    
    if 'owgr_rank_current' in predictions.columns:
        display_cols.append('owgr_rank_current')
        col_renames['owgr_rank_current'] = 'OWGR Rank'
    
    if 'tournament_rank' in predictions.columns:
        display_cols.append('tournament_rank')
        col_renames['tournament_rank'] = 'Actual Finish'
    
    # Create display DataFrame
    display_df = predictions[display_cols].head(10).copy()
    display_df = display_df.rename(columns=col_renames)
    
    print(display_df.to_string(index=False))
    
    # Show actual winner if available
    if 'tournament_rank' in predictions.columns:
        winners = predictions[predictions['tournament_rank'] == 1.0]
        if not winners.empty:
            winner = winners.iloc[0]
            winner_name = winner['name']
            winner_prob = winner['win_probability']
            winner_idx = predictions[predictions['name'] == winner_name].index[0]
            
            print(f"\n** Actual Winner: {winner_name}")
            print(f"   Model Rank: #{winner_idx + 1}")
            print(f"   Win Probability: {winner_prob*100:.2f}%")
    
    return predictions


def save_predictions(predictions, output_path=None):
    """Save predictions to CSV."""
    if output_path is None:
        output_path = MODEL_DIR / 'latest_predictions.csv'
    
    predictions.to_csv(output_path, index=False)
    print(f"\n[OK] Predictions saved to {output_path}")


if __name__ == '__main__':
    # Example: Predict Masters Tournament 2024
    tournament_name = "Masters Tournament"
    year = 2024
    
    try:
        predictions = predict_tournament(tournament_name, year)
        save_predictions(predictions)
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nAvailable tournaments:")
        
        # Show available tournaments
        data_file = DATA_DIR / 'espn_with_owgr_features.parquet'
        df = pd.read_parquet(data_file)
        tournaments = df[df['year'] == year]['tournament'].unique()
        for t in sorted(tournaments)[:10]:
            print(f"  - {t}")
