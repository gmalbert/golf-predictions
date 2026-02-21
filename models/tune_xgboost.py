"""
Optuna Hyperparameter Tuner for XGBoost

Performs time-series-aware cross-validation to find the best XGBoost
hyperparameters.  Best params are saved to models/saved_models/best_xgb_params.json
and can be passed directly to train_improved_model.py.

Usage:
    python models/tune_xgboost.py                    # 50 trials (default)
    python models/tune_xgboost.py --n-trials 200     # more thorough search
    python models/tune_xgboost.py --n-trials 10 --jobs 2  # fast dev run

See also: 04_models_and_features.md § Hyperparameter Optimisation
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

PARAMS_OUT = MODEL_DIR / "best_xgb_params.json"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_data():
    from models.train_improved_model import load_and_prepare_data, select_features
    print("[Optuna] Loading data …")
    df = load_and_prepare_data()
    feat_cols = select_features(df)
    # Use only training years for CV; hold test years completely out
    df = df[df["year"] <= 2024].copy()
    return df, feat_cols


def _tv_splits(df: pd.DataFrame, n_splits: int = 4):
    """
    Expanding-window time-series splits by year.
    Each fold keeps the final ~N months as the validation window.
    Returns [(train_idx, val_idx), …] using iloc-style integer indices.
    """
    years = sorted(df["year"].unique())
    if len(years) < n_splits + 1:
        raise ValueError(f"Not enough years ({len(years)}) for {n_splits} CV splits.")

    splits = []
    for i in range(n_splits):
        # Train on all years up to (years[-n_splits + i - 1])
        # Val  on the next year
        cutoff_year = years[-(n_splits - i)]
        train_mask = (df["year"] < cutoff_year).values
        val_mask   = (df["year"] == cutoff_year).values
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]
        if len(train_idx) and len(val_idx):
            splits.append((train_idx, val_idx))

    return splits


# ── Objective ─────────────────────────────────────────────────────────────────

def _objective(trial, df: pd.DataFrame, feat_cols: list[str]) -> float:
    import xgboost as xgb

    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
        "max_depth":          trial.suggest_int("max_depth", 2, 6),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    X = df[feat_cols].fillna(0).values
    y = (df["tournament_rank"] == 1).astype(int).values

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    splits = _tv_splits(df)
    aucs: list[float] = []

    for train_idx, val_idx in splits:
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=30,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
        proba = model.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], proba))

    return float(np.mean(aucs))


# ── Main ──────────────────────────────────────────────────────────────────────

def tune(n_trials: int = 50, n_jobs: int = 1) -> dict:
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is not installed. Run: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df, feat_cols = _load_data()
    print(f"[Optuna] {len(df):,} rows · {len(feat_cols)} features · {n_trials} trials")

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_golf",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    study.optimize(
        lambda trial: _objective(trial, df, feat_cols),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    best = study.best_params
    best_auc = study.best_value

    print(f"\n✅ Best CV AUC: {best_auc:.4f}")
    print("Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Save to JSON
    PARAMS_OUT.write_text(json.dumps({"best_auc": best_auc, "params": best}, indent=2))
    print(f"\n[OK] Saved → {PARAMS_OUT}")

    # Top-10 trials summary
    print("\nTop-10 trials:")
    trials_df = study.trials_dataframe(attrs=("number", "value", "params")).sort_values("value", ascending=False)
    print(trials_df.head(10).to_string(index=False))

    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for golf XGBoost model"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Parallel Optuna workers (default: 1)"
    )
    args = parser.parse_args()
    tune(n_trials=args.n_trials, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
