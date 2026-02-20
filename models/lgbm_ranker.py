"""
Tier 3 – LightGBM LambdaRank Model

Directly optimises ranking (NDCG) instead of binary classification.
Predicts *finish order* within each tournament rather than just win/loss.

Advantages over binary XGBoost:
  - Learns to separate 1st from 2nd, 5th from 15th, etc.
  - Better calibrated for outright-winner / top-5 value bets.
  - NDCG@5 is more betting-relevant than AUC.

Usage:
    python models/lgbm_ranker.py
    python models/lgbm_ranker.py --ndcg-at 5   # optimise NDCG@5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────
# Same pre-tournament features as improved model; ranking learns relative
# skill differences between players in the same field.

CANDIDATE_FEATURES = [
    "prior_avg_score", "prior_avg_score_5", "prior_avg_score_10",
    "prior_std_score", "prior_std_score_5", "prior_std_score_10",
    "prior_top10_rate_5", "prior_top10_rate_10",
    "prior_count",
    "last_event_score", "last_event_rank",
    "days_since_last_event",
    "career_best_rank",
    "tournaments_last_365d", "season_to_date_avg_score",
    "played_last_30d",
    "course_history_avg_score",
    "owgr_rank_current", "owgr_rank_4w_ago", "owgr_rank_12w_ago",
    "owgr_points_current",
    "owgr_rank_change_4w", "owgr_rank_change_12w", "owgr_rank_change_52w",
    "is_major", "is_playoff", "purse_tier",
    "course_type_enc", "grass_type_enc",
    "field_strength", "field_size",
    "sg_total_prev_season", "sg_putting_prev_season",
    "sg_approach_prev_season", "sg_off_tee_prev_season",
    "driving_distance_prev_season", "driving_accuracy_prev_season",
    "gir_pct_prev_season", "scoring_avg_prev_season",
    "sg_total_season", "sg_putting_season",
    "sg_approach_season", "sg_off_tee_season",
    "driving_accuracy_season", "gir_pct_season",
    "scoring_avg_season",
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
    raise FileNotFoundError("No feature parquet found. Run features/build_features.py first.")


def _select_features(df: pd.DataFrame) -> list[str]:
    available = [f for f in CANDIDATE_FEATURES if f in df.columns]
    missing = set(CANDIDATE_FEATURES) - set(available)
    print(f"[OK] Using {len(available)} features  ({len(missing)} unavailable)")
    return available


# ── Relevance scores ─────────────────────────────────────────────────────────
# LightGBM LambdaRank needs integer relevance labels where *higher = better*.
# We invert tournament_rank and clip to 0–4 (5 relevance levels).

def _make_relevance(rank_series: pd.Series) -> pd.Series:
    """
    Map numeric finish position to a 0-4 relevance score:
        1st       → 4
        2nd-3rd   → 3
        4th-10th  → 2
        11th-30th → 1
        31st+     → 0
    """
    rel = pd.Series(0, index=rank_series.index, dtype=int)
    rel[rank_series <= 30] = 1
    rel[rank_series <= 10] = 2
    rel[rank_series <= 3]  = 3
    rel[rank_series == 1]  = 4
    return rel


# ── Build LightGBM datasets ───────────────────────────────────────────────────

def _build_lgb_datasets(
    df: pd.DataFrame, feat_cols: list[str],
) -> tuple[lgb.Dataset, lgb.Dataset, list[str]]:
    """Return (train_data, val_data, feature_names)."""
    # Group key: unique tournament × year
    group_key = ["tournament_id", "year"] if "tournament_id" in df.columns \
                else ["tournament", "year"]

    # Drop rows with NaN in group-key columns (groupby silently excludes them,
    # which would cause a group-size / data-length mismatch in LightGBM).
    df = df.dropna(subset=group_key).copy()

    # Sort so groups are contiguous (required by LambdaRank).
    # Use only unique sort columns to avoid duplicate-column issues.
    sort_cols = list(dict.fromkeys(["year"] + group_key))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    X   = df[feat_cols].fillna(0)
    rel = _make_relevance(df["tournament_rank"])

    def _group_sizes(sub: pd.DataFrame) -> np.ndarray:
        """Return array of group sizes in row order (no sorting)."""
        return sub.groupby(group_key, sort=False).size().values

    # Time split
    train_mask = df["year"] <= 2022
    val_mask   = df["year"] >= 2023

    df_tr  = df[train_mask]
    df_val = df[val_mask]
    g_tr   = _group_sizes(df_tr)
    g_val  = _group_sizes(df_val)

    assert g_tr.sum()  == train_mask.sum(),  \
        f"Train group mismatch: {g_tr.sum()} vs {train_mask.sum()}"
    assert g_val.sum() == val_mask.sum(), \
        f"Val group mismatch: {g_val.sum()} vs {val_mask.sum()}"

    X_tr  = X[train_mask].values;   rel_tr  = rel[train_mask].values
    X_val = X[val_mask].values;     rel_val = rel[val_mask].values

    train_data = lgb.Dataset(X_tr,  rel_tr,  group=g_tr,  feature_name=feat_cols,
                             free_raw_data=False)
    val_data   = lgb.Dataset(X_val, rel_val, group=g_val, feature_name=feat_cols,
                             reference=train_data, free_raw_data=False)

    print(f"  Train: {len(X_tr):,} rows across {len(g_tr):,} tournaments")
    print(f"  Val:   {len(X_val):,} rows across {len(g_val):,} tournaments")

    return train_data, val_data, feat_cols


# ── Training ─────────────────────────────────────────────────────────────────

def train(ndcg_at: int = 5) -> lgb.Booster:
    print("\n" + "=" * 60)
    print(f"LIGHTGBM LAMBDARANK  (optimising NDCG@{ndcg_at})")
    print("=" * 60)

    df = _load_data()
    feat_cols = _select_features(df)

    if "tournament_rank" not in df.columns:
        raise KeyError("'tournament_rank' column is required.")

    train_data, val_data, feat_cols = _build_lgb_datasets(df, feat_cols)

    params = {
        "objective":    "lambdarank",
        "metric":       "ndcg",
        "ndcg_eval_at": [1, 5, 10],
        "learning_rate": 0.05,
        "num_leaves":    63,
        "min_data_in_leaf": 20,
        "lambda_l1":    0.1,
        "lambda_l2":    1.0,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":      -1,
        "seed":         42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ]

    print(f"\nTraining LambdaRank …")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    best = model.best_score.get("valid_0", {})
    print(f"\nBest scores on validation set:")
    for metric, value in sorted(best.items()):
        print(f"  {metric}: {value:.4f}")

    # ── Feature importance ────────────────────────────────────────────────
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=feat_cols,
    ).sort_values(ascending=False)
    print(f"\nTop 15 features by gain:")
    for feat, imp in importance.head(15).items():
        print(f"  {feat:<45s}  {imp:.1f}")

    # ── Save ─────────────────────────────────────────────────────────────
    model_path   = MODEL_DIR / "lgbm_ranker.txt"
    feat_path    = MODEL_DIR / "lgbm_ranker_features.txt"
    import_path  = MODEL_DIR / "lgbm_ranker_importance.csv"

    model.save_model(str(model_path))
    feat_path.write_text("\n".join(feat_cols))
    importance.to_csv(str(import_path))

    print(f"\n[OK] Model saved:      {model_path}")
    print(f"[OK] Features saved:   {feat_path}")
    print(f"[OK] Importance saved: {import_path}")

    return model


# ── Prediction helper ─────────────────────────────────────────────────────────

def predict_field(df_upcoming: pd.DataFrame) -> pd.DataFrame:
    """
    Score a field of players for an upcoming tournament.

    Parameters
    ----------
    df_upcoming : DataFrame with the same feature columns used at training.

    Returns
    -------
    DataFrame with 'rank_score' column; higher score = model thinks better finish.
    """
    model_path = MODEL_DIR / "lgbm_ranker.txt"
    feat_path  = MODEL_DIR / "lgbm_ranker_features.txt"

    if not model_path.exists():
        raise FileNotFoundError("lgbm_ranker.txt not found — run models/lgbm_ranker.py first.")

    model     = lgb.Booster(model_file=str(model_path))
    feat_cols = feat_path.read_text().splitlines()

    X = df_upcoming[[c for c in feat_cols if c in df_upcoming.columns]].fillna(0)
    # Pad missing columns with 0
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_cols]

    df_upcoming = df_upcoming.copy()
    df_upcoming["rank_score"] = model.predict(X)
    df_upcoming = df_upcoming.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df_upcoming["predicted_rank"] = range(1, len(df_upcoming) + 1)
    return df_upcoming


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaRank model.")
    parser.add_argument("--ndcg-at", type=int, default=5,
                        help="Primary NDCG cut-off to optimise (default: 5)")
    args = parser.parse_args()
    train(ndcg_at=args.ndcg_at)


if __name__ == "__main__":
    main()
