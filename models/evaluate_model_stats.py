"""
Generate comprehensive model quality statistics as dataframes.

Usage:
    python models/evaluate_model_stats.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

# Paths
PROJ_ROOT = Path(__file__).resolve().parent.parent
# Prefer the extended-features parquet (has SG stats, weather, course info, etc.);
# fall back to the lighter OWGR-only parquet if it's absent.
_EXT = PROJ_ROOT / 'data_files' / 'espn_with_extended_features.parquet'
_OWGR = PROJ_ROOT / 'data_files' / 'espn_with_owgr_features.parquet'
DATA_PATH = _EXT if _EXT.exists() else _OWGR
MODEL_DIR = PROJ_ROOT / 'models' / 'saved_models'
MODEL_PATH = MODEL_DIR / 'winner_predictor_v2.json'
FEATURES_PATH = MODEL_DIR / 'model_features_v2.txt'


def load_model_and_features():
    """Load the trained XGBoost model and feature columns."""
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))
    
    with open(FEATURES_PATH, 'r') as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    
    return model, feature_cols


def load_and_prepare_data(feature_cols):
    """Load and split data into train/val/test sets (same as training)."""
    df = pd.read_parquet(DATA_PATH)
    
    # Remove records without OWGR features
    df = df.dropna(subset=['owgr_points_current', 'owgr_rank_current'])
    
    # Add tournament_datetime if missing
    if 'tournament_datetime' not in df.columns:
        df['tournament_datetime'] = pd.to_datetime(df['date'])
    
    # Add finish_position for target (using tournament_rank)
    if 'finish_position' not in df.columns:
        df['finish_position'] = df['tournament_rank']
    
    # Create target
    df['won'] = (df['finish_position'] == 1).astype(int)
    
    # Time-based split (same as training)
    df = df.sort_values('tournament_datetime')
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df, feature_cols


def get_predictions(model, df, feature_cols):
    """Get model predictions for a dataset."""
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Convert to DMatrix
    dmatrix = xgb.DMatrix(X)
    
    # Get probabilities
    proba = model.predict(dmatrix)
    
    return proba, df['won'].values


def compute_overall_metrics(y_true, y_pred_proba):
    """Compute overall classification metrics."""
    metrics = {
        'AUC-ROC': roc_auc_score(y_true, y_pred_proba),
        'Log Loss': log_loss(y_true, y_pred_proba),
        'Average Precision': average_precision_score(y_true, y_pred_proba),
        'Brier Score': brier_score_loss(y_true, y_pred_proba),
    }
    
    # Add counts
    metrics['Total Samples'] = len(y_true)
    metrics['Positive Samples'] = int(y_true.sum())
    metrics['Negative Samples'] = int(len(y_true) - y_true.sum())
    metrics['Class Imbalance Ratio'] = y_true.sum() / len(y_true)
    
    return metrics


def compute_threshold_metrics(y_true, y_pred_proba, thresholds=[0.01, 0.05, 0.1, 0.2, 0.5]):
    """Compute precision, recall, F1 at different probability thresholds."""
    results = []
    
    for thresh in thresholds:
        y_pred_binary = (y_pred_proba >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Threshold': thresh,
            'True Positives': int(tp),
            'False Positives': int(fp),
            'True Negatives': int(tn),
            'False Negatives': int(fn),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Predicted Positive': int(tp + fp)
        })
    
    return pd.DataFrame(results)


def compute_calibration_bins(y_true, y_pred_proba, n_bins=10):
    """Compute calibration statistics by binning predicted probabilities."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='quantile')
    
    # Also compute ECE (Expected Calibration Error)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_mean_pred = y_pred_proba[mask].mean()
            bin_mean_true = y_true[mask].mean()
            bin_count = mask.sum()
            bin_error = abs(bin_mean_pred - bin_mean_true)
            
            calibration_data.append({
                'Bin': i + 1,
                'Count': int(bin_count),
                'Mean Predicted Prob': bin_mean_pred,
                'Actual Win Rate': bin_mean_true,
                'Calibration Error': bin_error
            })
    
    cal_df = pd.DataFrame(calibration_data)
    
    # Compute ECE
    ece = (cal_df['Calibration Error'] * cal_df['Count']).sum() / cal_df['Count'].sum()
    
    return cal_df, ece


def compute_top_n_accuracy(df, y_pred_proba, feature_cols, top_ns=[1, 5, 10, 20, 50]):
    """
    Compute top-N accuracy: for each tournament, does the actual winner
    appear in the top N predicted players?
    """
    results = []
    
    # Add predictions to dataframe
    pred_df = df.copy()
    pred_df['win_probability'] = y_pred_proba
    
    # Group by tournament (use tournament_id and date)
    tournaments = pred_df.groupby(['tournament_id', 'tournament', 'tournament_datetime'])
    
    for top_n in top_ns:
        correct_count = 0
        total_tournaments = 0
        
        for (tid, tname, tdate), group in tournaments:
            # Sort by predicted probability
            sorted_group = group.sort_values('win_probability', ascending=False)
            
            # Get top N
            top_n_players = sorted_group.head(top_n)
            
            # Check if actual winner is in top N
            actual_winner = group[group['won'] == 1]
            
            if len(actual_winner) > 0:
                total_tournaments += 1
                if actual_winner.index[0] in top_n_players.index:
                    correct_count += 1
        
        accuracy = correct_count / total_tournaments if total_tournaments > 0 else 0
        
        results.append({
            'Top N': top_n,
            'Tournaments': total_tournaments,
            'Correct Predictions': correct_count,
            'Top-N Accuracy': accuracy
        })
    
    return pd.DataFrame(results)


def get_feature_importance(model, feature_cols):
    """Extract feature importance from the model."""
    importance_dict = model.get_score(importance_type='gain')
    
    # Map feature names (model uses fN format)
    feature_map = {f'f{i}': name for i, name in enumerate(feature_cols)}
    
    importance_data = []
    for feat_id, importance in importance_dict.items():
        feat_name = feature_map.get(feat_id, feat_id)
        importance_data.append({
            'Feature': feat_name,
            'Importance (Gain)': importance
        })
    
    importance_df = pd.DataFrame(importance_data).sort_values('Importance (Gain)', ascending=False)
    
    # Normalize to percentage
    total_importance = importance_df['Importance (Gain)'].sum()
    importance_df['Importance (%)'] = (importance_df['Importance (Gain)'] / total_importance * 100)
    importance_df['Cumulative (%)'] = importance_df['Importance (%)'].cumsum()
    
    return importance_df


def main():
    """Generate all model quality statistics."""
    print("="*80)
    print("MODEL QUALITY STATISTICS")
    print("="*80)
    
    # Load model and data
    print("\nLoading model and data...")
    model, feature_cols = load_model_and_features()
    train_df, val_df, test_df, feature_cols = load_and_prepare_data(feature_cols)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Val samples: {len(val_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    
    # Get predictions for val + test only.
    # Train metrics are intentionally excluded — a model always scores higher
    # on its own training data, making those numbers misleading.
    print("\nGenerating predictions...")
    val_proba, y_val = get_predictions(model, val_df, feature_cols)
    test_proba, y_test = get_predictions(model, test_df, feature_cols)
    
    # ========================================================================
    # 1. Overall Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("1. OVERALL METRICS (out-of-sample only)")
    print("="*80)
    
    overall_metrics = pd.DataFrame({
        'Validation': compute_overall_metrics(y_val, val_proba),
        'Test': compute_overall_metrics(y_test, test_proba)
    }).T
    
    print("\n", overall_metrics.to_string())
    
    # ========================================================================
    # 2. Threshold-based Metrics (Test Set)
    # ========================================================================
    print("\n" + "="*80)
    print("2. THRESHOLD-BASED METRICS (Test Set)")
    print("="*80)
    
    threshold_metrics = compute_threshold_metrics(y_test, test_proba)
    print("\n", threshold_metrics.to_string(index=False))
    
    # ========================================================================
    # 3. Calibration Analysis (Test Set)
    # ========================================================================
    print("\n" + "="*80)
    print("3. CALIBRATION ANALYSIS (Test Set)")
    print("="*80)
    
    calibration_df, ece = compute_calibration_bins(y_test, test_proba, n_bins=10)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("\n", calibration_df.to_string(index=False))
    
    # ========================================================================
    # 4. Top-N Accuracy (Test Set)
    # ========================================================================
    print("\n" + "="*80)
    print("4. TOP-N ACCURACY (Test Set)")
    print("="*80)
    print("Does the actual winner appear in the top N predicted players?")
    
    top_n_accuracy = compute_top_n_accuracy(test_df, test_proba, feature_cols)
    print("\n", top_n_accuracy.to_string(index=False))
    
    # ========================================================================
    # 5. Feature Importance
    # ========================================================================
    print("\n" + "="*80)
    print("5. FEATURE IMPORTANCE (Top 20)")
    print("="*80)
    
    importance_df = get_feature_importance(model, feature_cols)
    print("\n", importance_df.head(20).to_string(index=False))
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
Primary Metric: AUC-ROC
  - Train: {overall_metrics.loc['Train', 'AUC-ROC']:.4f}
  - Validation: {overall_metrics.loc['Validation', 'AUC-ROC']:.4f}
  - Test: {overall_metrics.loc['Test', 'AUC-ROC']:.4f}

Calibration: ECE = {ece:.4f} (lower is better; <0.1 is well-calibrated)

Top-5 Accuracy: {top_n_accuracy[top_n_accuracy['Top N']==5]['Top-N Accuracy'].values[0]:.2%}
  (Winner appears in model's top 5 predictions)

Top-10 Accuracy: {top_n_accuracy[top_n_accuracy['Top N']==10]['Top-N Accuracy'].values[0]:.2%}
  (Winner appears in model's top 10 predictions)
""")
    
    print("="*80)
    
    # Return all dataframes for programmatic access
    return {
        'overall_metrics': overall_metrics,
        'threshold_metrics': threshold_metrics,
        'calibration': calibration_df,
        'calibration_ece': ece,
        'top_n_accuracy': top_n_accuracy,
        'feature_importance': importance_df
    }


if __name__ == '__main__':
    stats = main()
