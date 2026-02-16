"""
Baseline Tournament Winner Prediction Model

This model predicts tournament winners using:
- Historical performance features
- OWGR (world ranking) features
- Recent form metrics

Model: XGBoost Classifier (binary: win/loss)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import xgboost as xgb
import joblib


DATA_DIR = Path(__file__).parent.parent / 'data_files'
MODEL_DIR = Path(__file__).parent / 'saved_models'
MODEL_DIR.mkdir(exist_ok=True)


def load_training_data():
    """Load OWGR-enhanced features for model training."""
    print("\n" + "="*70)
    print("LOADING TRAINING DATA")
    print("="*70)
    
    features_file = DATA_DIR / 'espn_with_owgr_features.parquet'
    
    if not features_file.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_file}\n"
            f"Run: python features/build_owgr_features.py"
        )
    
    df = pd.read_parquet(features_file)
    print(f"\n[OK] Loaded {len(df):,} tournament records")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Players: {df['player_id'].nunique():,}")
    print(f"  Tournaments: {df['tournament'].nunique():,}")
    
    return df


def prepare_features(df):
    """
    Prepare features and target variable for training.
    
    Creates binary classification target: 1 = won tournament, 0 = did not win
    """
    # Create target variable: did they finish 1st?
    df['won_tournament'] = (df['tournament_rank'] == 1).astype(int)
    
    wins = df['won_tournament'].sum()
    non_wins = len(df) - wins
    pct_win = (wins / len(df)) * 100
    pct_non_win = (non_wins / len(df)) * 100
    
    print(f"\n  Wins: {wins:,} ({pct_win:.2f}%)")
    print(f"  Non-wins: {non_wins:,} ({pct_non_win:.2f}%)\n")
    print(f"\n>> Target distribution:")
    print(df['won_tournament'].value_counts())
    
    # Select features
    # Historical performance
    historical_features = [
        'prior_avg_score',
        'prior_avg_score_5',
        'prior_avg_score_10',
        'prior_best_finish',
        'prior_worst_finish',
        'prior_tournaments',
    ]
    
    # Recent form
    recent_form_features = [
        'tournaments_last_30d',
        'tournaments_last_90d',
        'tournaments_last_365d',
        'avg_score_last_30d',
        'avg_score_last_90d',
        'avg_score_last_365d',
        'best_finish_last_30d',
        'best_finish_last_90d',
        'best_finish_last_365d',
    ]
    
    # OWGR features
    owgr_features = [
        'owgr_rank_current',
        'owgr_points_current',
        'owgr_rank_change',
        'owgr_momentum_4wk',
        'owgr_momentum_12wk',
        'owgr_momentum_24wk',
        'owgr_staleness_days',
        'owgr_rank_percentile',
        'owgr_points_percentile',
    ]
    
    selected_features = historical_features + recent_form_features + owgr_features
    
    # Check which features are available
    available_cols = [col for col in selected_features if col in df.columns]
    print(f"\n[OK] Using {len(available_cols)} features:")
    print("  " + ", ".join(available_cols[:5]) + "...")
    
    missing = set(selected_features) - set(available_cols)
    if missing:
        print(f"\n  Warning: Missing features: {missing}")
    
    return df, available_cols


def train_model(df, feature_cols):
    """Train XGBoost classifier."""
    print("\n" + "="*70)
    print("TRAINING MODEL")
    print("="*70)
    
    # Prepare features and target
    X = df[feature_cols].fillna(0)
    y = df['won_tournament']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n>> Data split:")
    print(f"  Train: {len(X_train):,} records ({y_train.sum():,} wins)")
    print(f"  Test:  {len(X_test):,} records ({y_test.sum():,} wins)")
    
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n  Class imbalance ratio: {scale_pos_weight:.1f}:1 (non-win:win)")
    print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Train XGBoost model
    print("\n  Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    
    print(f"\n>> Model Performance:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Test AUC:  {test_auc:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n>> Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']*100:6.2f}%")
    
    return model, importance_df


def save_model(model, feature_cols, importance_df):
    """Save trained model and metadata."""
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    # Save model
    model_path = MODEL_DIR / 'baseline_winner_predictor.joblib'
    joblib.dump(model, model_path)
    print(f"\n[OK] Model saved: {model_path}")
    
    # Save feature list
    features_path = MODEL_DIR / 'model_features.txt'
    features_path.write_text('\n'.join(feature_cols))
    print(f"[OK] Features saved: {features_path}")
    
    # Save feature importance
    importance_path = MODEL_DIR / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"[OK] Importance saved: {importance_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)


def main():
    """Main training pipeline."""
    # Load data
    df = load_training_data()
    
    # Prepare features
    df, feature_cols = prepare_features(df)
    
    # Train model
    model, importance_df = train_model(df, feature_cols)
    
    # Save model
    save_model(model, feature_cols, importance_df)


if __name__ == '__main__':
    main()
