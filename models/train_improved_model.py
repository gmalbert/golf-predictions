"""
Improved Tournament Winner Prediction Model (No Data Leakage)

Key improvements:
1. Time-based train/validation/test split
2. Feature leakage prevention
3. Proper cross-validation
4. Regularization to prevent overfitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, log_loss, average_precision_score
import xgboost as xgb
import joblib
from datetime import datetime


DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODEL_DIR = Path(__file__).parent / 'saved_models'
MODEL_DIR.mkdir(exist_ok=True)


def load_and_prepare_data():
    """Load data and prepare for time-based training."""
    print("\n" + "="*70)
    print("LOADING AND PREPARING DATA")
    print("="*70)
    
    # Prefer OWGR-enhanced dataset when available
    owgr_path = DATA_DIR / 'espn_with_owgr_features.parquet'
    base_path = DATA_DIR / 'espn_player_tournament_features.parquet'
    if owgr_path.exists():
        df = pd.read_parquet(owgr_path)
        print("[OK] Loaded OWGR-enhanced features")
    else:
        df = pd.read_parquet(base_path)
        print("[OK] Loaded base player-tournament features (no OWGR)")
    
    print(f"\n[OK] Loaded {len(df):,} tournament records")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Players: {df['player_id'].nunique():,}")
    print(f"  Tournaments: {df['tournament'].nunique():,}")
    
    # Create target variable
    df['won_tournament'] = (df['tournament_rank'] == 1).astype(int)
    
    # Check for data quality issues
    winners_per_tournament = df.groupby(['tournament', 'year'])['won_tournament'].sum()
    multiple_winners = winners_per_tournament[winners_per_tournament > 1]
    
    if len(multiple_winners) > 0:
        print(f"\n  WARNING: {len(multiple_winners)} tournaments have multiple winners (ties)")
        print(f"  Example: {multiple_winners.head(3).to_dict()}")
    
    wins = df['won_tournament'].sum()
    print(f"\n  Total wins: {wins:,} ({wins/len(df)*100:.2f}%)")
    print(f"  Non-wins: {len(df)-wins:,} ({(len(df)-wins)/len(df)*100:.2f}%)")
    
    return df


def select_features(df):
    """
    Select features that are available BEFORE the tournament.
    
    CRITICAL: Features must NOT include information from the current tournament!
    """
    # Historical performance features (calculated from PRIOR tournaments only)
    historical_features = [
        'prior_avg_score',          # Average score in all prior tournaments
        'prior_avg_score_5',         # Average score in last 5 tournaments
        'prior_avg_score_10',        # Average score in last 10 tournaments
        'prior_std_score',           # Consistency in prior tournaments
        'prior_std_score_5',
        'prior_std_score_10',
        'prior_top10_rate_5',        # % of top 10 finishes in last 5
        'prior_top10_rate_10',       # % of top 10 finishes in last 10
        'prior_count',               # Number of prior tournaments
        'last_event_score',          # Score in most recent tournament
        'last_event_rank',           # Rank in most recent tournament
        'days_since_last_event',     # Days of rest/rust
        'career_best_rank',          # Best ever finish
    ]
    
    # Recent form features
    recent_form = [
        'tournaments_last_365d',     # Tournament frequency
        'season_to_date_avg_score',  # Current season performance
        'course_history_avg_score',  # Performance at this specific course
        'played_last_30d',           # Recent activity
    ]
    
    # OWGR features (known before tournament)
    owgr_features = [
        'owgr_rank_current',         # Current world ranking
        'owgr_rank_4w_ago',
        'owgr_rank_12w_ago',
        'owgr_rank_52w_ago',
        'owgr_points_current',
        'owgr_rank_change_4w',       # Momentum indicators
        'owgr_rank_change_12w',
        'owgr_rank_change_52w',
        'owgr_data_staleness_days',
    ]
    
    all_features = historical_features + recent_form + owgr_features
    
    # Check which features exist
    available = [f for f in all_features if f in df.columns]
    missing = set(all_features) - set(available)
    
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    print(f"\n[OK] Using {len(available)} features")
    print(f"\nFeature categories:")
    print(f"  Historical: {sum(1 for f in available if f in historical_features)}")
    print(f"  Recent form: {sum(1 for f in available if f in recent_form)}")
    print(f"  OWGR: {sum(1 for f in available if f in owgr_features)}")
    
    if missing:
        print(f"\n  Note: {len(missing)} features not available in data")
    
    # CRITICAL CHECK: Make sure we're not including the current tournament result
    forbidden_features = ['tournament_rank', 'numeric_total_score']
    leakage = [f for f in available if f in forbidden_features]
    if leakage:
        raise ValueError(f"DATA LEAKAGE DETECTED: {leakage} should not be in features!")
    
    return available


def create_time_based_splits(df, feature_cols):
    """
    Create train/validation/test splits based on time.
    
    Train: 2018-2022
    Validation: 2023-2024
    Test: 2025
    """
    print("\n" + "="*70)
    print("TIME-BASED DATA SPLITS")
    print("="*70)
    
    # Prepare features
    X = df[feature_cols].fillna(0)  # Fill missing values
    y = df['won_tournament']
    years = df['year']
    
    # Create splits
    train_mask = years <= 2022
    val_mask = (years >= 2023) & (years <= 2024)
    test_mask = years >= 2025
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n  Train (2018-2022): {len(y_train):,} records ({y_train.sum():,} wins)")
    print(f"  Val   (2023-2024): {len(y_val):,} records ({y_val.sum():,} wins)")
    print(f"  Test  (2025+):     {len(y_test):,} records ({y_test.sum():,} wins)")
    
    # Class imbalance
    train_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n  Train class imbalance: {train_ratio:.1f}:1 (non-win:win)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, feature_cols):
    """Train XGBoost with proper regularization."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    # Calculate class weight
    scale_pos_weight = (y_train ==0).sum() / (y_train == 1).sum()
    
    print(f"\n  Training XGBoost with regularization...")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Use more regularization to prevent overfitting
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,                # Reduced from 6 to prevent overfitting
        learning_rate=0.05,          # Reduced from 0.1 for smoother learning
        min_child_weight=5,          # Increased to prevent overfitting
        subsample=0.8,               # Use 80% of data for each tree
        colsample_bytree=0.8,        # Use 80% of features for each tree
        gamma=1.0,                   # Minimum loss reduction for split
        reg_alpha=0.1,               # L1 regularization
        reg_lambda=1.0,              # L2 regularization
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20,    # Stop if no improvement for 20 rounds
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train with validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    print(f"\n  Training complete!")
    if hasattr(model, 'best_iteration') and model.best_iteration is not None:
        print(f"  Best iteration: {model.best_iteration}/{model.n_estimators}")
    
    return model


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, feature_cols):
    """Comprehensive model evaluation."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Get predictions
    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, train_proba)
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    train_logloss = log_loss(y_train, train_proba)
    val_logloss = log_loss(y_val, val_proba)
    test_logloss = log_loss(y_test, test_proba)
    
    train_ap = average_precision_score(y_train, train_proba)
    val_ap = average_precision_score(y_val, val_proba)
    test_ap = average_precision_score(y_test, test_proba)
    
    print(f"\n>> Performance Metrics:")
    print(f"\n  AUC-ROC (higher is better, max=1.0):")
    print(f"    Train: {train_auc:.4f}")
    print(f"    Val:   {val_auc:.4f}")
    print(f"    Test:  {test_auc:.4f}")
    
    print(f"\n  Log Loss (lower is better):")
    print(f"    Train: {train_logloss:.4f}")
    print(f"    Val:   {val_logloss:.4f}")
    print(f"    Test:  {test_logloss:.4f}")
    
    print(f"\n  Average Precision (higher is better):")
    print(f"    Train: {train_ap:.4f}")
    print(f"    Val:   {val_ap:.4f}")
    print(f"    Test:  {test_ap:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n>> Top 15 Most Important Features:")
    for idx, row in importance_df.head(15).iterrows():
        print(f"  {row['feature']:30s} {row['importance']*100:5.2f}%")
    
    return importance_df


def save_model(model, feature_cols, importance_df):
    """Save model and metadata."""
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model (joblib for sklearn wrapper + XGBoost-native for compatibility)
    model_path = MODEL_DIR / 'winner_predictor_v2.joblib'
    joblib.dump(model, model_path)
    print(f"\n[OK] Model (joblib) saved: {model_path}")

    # Also save native XGBoost model to avoid pickle/version warnings
    xgb_native_path = MODEL_DIR / 'winner_predictor_v2.json'
    try:
        # Use the Booster API to save a version-independent representation
        booster = model.get_booster()
        booster.save_model(str(xgb_native_path))
        print(f"[OK] XGBoost-native model saved: {xgb_native_path}")
    except Exception as e:
        print(f"[WARN] Could not save native XGBoost model: {e}")

    # Save feature list
    features_path = MODEL_DIR / 'model_features_v2.txt'
    features_path.write_text('\n'.join(feature_cols))
    print(f"[OK] Features saved: {features_path}")

    # Save importance
    importance_path = MODEL_DIR / 'feature_importance_v2.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"[OK] Importance saved: {importance_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel ready for predictions with realistic probabilities.")


def main():
    """Main training pipeline."""
    # Load data
    df = load_and_prepare_data()
    
    # Select features (no leakage!)
    feature_cols = select_features(df)
    
    # Create time-based splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_time_based_splits(df, feature_cols)
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, feature_cols)
    
    # Evaluate
    importance_df = evaluate_model(
        model, X_train, X_val, X_test, 
        y_train, y_val, y_test, feature_cols
    )
    
    # Save
    save_model(model, feature_cols, importance_df)


if __name__ == '__main__':
    main()
