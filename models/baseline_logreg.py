"""
Tier 1 – Baseline Logistic Regression Model

Predicts whether a player will finish Top-10 (binary).

Why logistic regression first?
  - Forces you to think about scale / multicollinearity.
  - Coefficients are directly interpretable.
  - Fast to train; provides a hard floor for XGBoost to beat.

Usage:
    python models/baseline_logreg.py
    python models/baseline_logreg.py --top 5   # predict Top-5 instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────
# These are all pre-tournament features present in the current data pipeline.
# Logistic regression requires numeric, finite values – missing are filled with 0.

CANDIDATE_FEATURES = [
    # Historical form
    "prior_avg_score", "prior_avg_score_5", "prior_avg_score_10",
    "prior_std_score", "prior_std_score_5", "prior_std_score_10",
    "prior_top10_rate_5", "prior_top10_rate_10",
    "prior_count",
    # Most-recent event
    "last_event_score", "last_event_rank",
    "days_since_last_event",
    "career_best_rank",
    # Season momentum
    "tournaments_last_365d", "season_to_date_avg_score",
    "played_last_30d",
    # Course-specific
    "course_history_avg_score",
    # OWGR
    "owgr_rank_current", "owgr_rank_4w_ago", "owgr_rank_12w_ago",
    "owgr_points_current",
    "owgr_rank_change_4w", "owgr_rank_change_12w",
    # Extended context (not always present)
    "is_major", "is_playoff", "purse_tier", "purse_size_m",
    "course_type_enc", "grass_type_enc", "course_yardage",
    "field_strength", "field_size",
    "course_length_fit", "grass_fit",
    # SG previous season (zero leakage)
    "sg_total_prev_season", "sg_putting_prev_season",
    "sg_approach_prev_season", "sg_off_tee_prev_season",
    "driving_distance_prev_season",
    "driving_accuracy_prev_season",
    "gir_pct_prev_season", "scoring_avg_prev_season",
]

# ── Data loading ─────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    for candidate in [
        DATA_DIR / "espn_with_extended_features.parquet",
        DATA_DIR / "espn_with_owgr_features.parquet",
        DATA_DIR / "espn_player_tournament_features.parquet",
    ]:
        if candidate.exists():
            df = pd.read_parquet(candidate)
            print(f"[OK] Loaded {len(df):,} rows from {candidate.name}")
            return df
    raise FileNotFoundError(
        "No feature parquet found in data_files/. "
        "Run features/build_features.py first."
    )


def _build_target(df: pd.DataFrame, top_n: int) -> pd.Series:
    """Binary: did the player finish in the top N?"""
    if "tournament_rank" not in df.columns:
        raise KeyError("'tournament_rank' column is required.")
    return (df["tournament_rank"] <= top_n).astype(int)


def _select_features(df: pd.DataFrame) -> list[str]:
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    missing = set(CANDIDATE_FEATURES) - set(available)
    print(f"[OK] Using {len(available)} features  ({len(missing)} unavailable)")
    if missing:
        print(f"     Missing: {sorted(missing)[:8]}{'…' if len(missing) > 8 else ''}")
    return available


# ── Training ─────────────────────────────────────────────────────────────────

def _time_split(df: pd.DataFrame):
    train = df["year"] <= 2022
    val   = (df["year"] >= 2023) & (df["year"] <= 2024)
    test  = df["year"] >= 2025
    print(f"  Train (≤2022):      {train.sum():,} rows")
    print(f"  Validation (2023-24): {val.sum():,} rows")
    print(f"  Test (2025+):       {test.sum():,} rows")
    return train, val, test


def train(top_n: int = 10) -> Pipeline:
    print("\n" + "=" * 60)
    print(f"LOGISTIC REGRESSION BASELINE  (Top-{top_n})")
    print("=" * 60)

    df = _load_data()
    feat_cols = _select_features(df)
    y = _build_target(df, top_n)

    # Guard against leakage
    for forbidden in ["tournament_rank", "numeric_total_score"]:
        if forbidden in feat_cols:
            feat_cols.remove(forbidden)
            print(f"  [WARN] Removed leaking column: {forbidden}")

    X = df[feat_cols].fillna(0)

    # ── TimeSeriesSplit CV on train data ──────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    # Sort by year so splits are time-ordered
    df_sorted = df.reset_index(drop=True).sort_values("year")
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y.loc[df_sorted.index]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            C=0.1, max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )),
    ])

    print("\nCross-validation (TimeSeriesSplit, 5 folds):")
    fold_aucs: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
        X_tr, X_val = X_sorted.iloc[tr_idx], X_sorted.iloc[val_idx]
        y_tr, y_val = y_sorted.iloc[tr_idx], y_sorted.iloc[val_idx]
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        fold_aucs.append(auc)
        print(f"  Fold {fold}  AUC: {auc:.4f}")

    print(f"\nCV Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    # ── Final train + eval ────────────────────────────────────────────────
    df_sorted_year = df.sort_values("year").reset_index(drop=True)
    X_all = df[feat_cols].fillna(0)
    y_all = y

    train_mask, val_mask, test_mask = _time_split(df)

    pipe.fit(X_all[train_mask], y_all[train_mask])

    for split_name, mask in [("Val (2023-24)", val_mask), ("Test (2025+)", test_mask)]:
        if mask.sum() == 0:
            print(f"  {split_name}: no data")
            continue
        probs = pipe.predict_proba(X_all[mask])[:, 1]
        auc = roc_auc_score(y_all[mask], probs)
        ap  = average_precision_score(y_all[mask], probs)
        ll  = log_loss(y_all[mask], probs)
        print(f"\n  {split_name}:")
        print(f"    AUC-ROC:  {auc:.4f}")
        print(f"    Avg Prec: {ap:.4f}")
        print(f"    Log-Loss: {ll:.4f}")

    # ── Coefficients ─────────────────────────────────────────────────────
    coef = pd.Series(
        pipe["logreg"].coef_[0], index=feat_cols
    ).sort_values(key=abs, ascending=False)
    print("\nTop 10 coefficients by magnitude:")
    for feat, c in coef.head(10).items():
        direction = "↑" if c > 0 else "↓"
        print(f"  {direction}  {feat:<40s}  {c:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = MODEL_DIR / f"logreg_top{top_n}.joblib"
    feat_path = MODEL_DIR / f"logreg_top{top_n}_features.txt"
    joblib.dump(pipe, save_path)
    feat_path.write_text("\n".join(feat_cols))
    print(f"\n[OK] Model saved:    {save_path}")
    print(f"[OK] Features saved: {feat_path}")

    return pipe


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train logistic regression baseline.")
    parser.add_argument("--top", type=int, default=10,
                        help="Top-N finish to predict (default: 10)")
    args = parser.parse_args()
    train(top_n=args.top)


if __name__ == "__main__":
    main()
