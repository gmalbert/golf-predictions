"""
Stacked Ensemble Model

Combines LogReg, XGBoost, and LightGBM predictions using a simple meta-learner
(logistic regression on base-model outputs).  Falls back to a weighted average
when only a subset of base models are available.

Usage:
    python models/ensemble.py              # train and save
    python models/ensemble.py --predict    # load saved model and print probabilities

See also: 04_models_and_features.md § Ensemble
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2022].copy()
    val   = df[(df["year"] >= 2023) & (df["year"] <= 2024)].copy()
    test  = df[df["year"] >= 2025].copy()
    return train, val, test


def _target(df: pd.DataFrame) -> pd.Series:
    return (df["tournament_rank"] == 1).astype(int)


# ── Load base models ───────────────────────────────────────────────────────────

def _load_base_models() -> dict:
    """Load any of the three trained base models that exist on disk."""
    candidates = {
        "logreg":  MODEL_DIR / "logreg_top1.joblib",
        "xgboost": MODEL_DIR / "winner_predictor_v2.joblib",
        "lgbm":    MODEL_DIR / "lgbm_ranker.pkl",
    }
    loaded = {}
    for name, path in candidates.items():
        if path.exists():
            loaded[name] = joblib.load(path)
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [--] {name}: not found ({path.name})")
    return loaded


def _get_base_features(df: pd.DataFrame) -> list[str]:
    """Features shared by all base models (intersection)."""
    from models.train_improved_model import select_features
    return select_features(df)


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Safely get probability for class 1 regardless of model type."""
    try:
        return model.predict_proba(X.fillna(0))[:, 1]
    except AttributeError:
        # LightGBM Booster returns raw scores
        return model.predict(X.fillna(0).values)


# ── Ensemble ──────────────────────────────────────────────────────────────────

class GolfEnsemble:
    """
    Meta-learner that stacks base model predictions.

    If ≥2 base models are available a logistic regression meta-model is trained
    on their outputs.  With only one base model available the ensemble degrades
    gracefully to that model.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self.base_models: dict = {}
        self.feature_cols: list[str] = []
        self.meta_model: LogisticRegression | None = None
        self.scaler = StandardScaler()
        # Optional manual weights {model_name: weight}; used when meta_model is None
        self.weights = weights

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "GolfEnsemble":
        """Train base models (if not pre-loaded) and fit the meta-model on val set."""
        print("\n[Ensemble] Loading base models …")
        self.base_models = _load_base_models()

        if not self.base_models:
            raise RuntimeError("No base models found. Train at least one base model first.")

        self.feature_cols = _get_base_features(train_df)
        X_val = train_df[self.feature_cols].fillna(0)
        y_val = _target(val_df)
        X_val2 = val_df[self.feature_cols].fillna(0)

        # Build meta-features from validation predictions (out-of-fold style)
        meta_X = self._base_predictions(X_val2)

        if meta_X.shape[1] >= 2:
            print("\n[Ensemble] Fitting meta-model (LogReg on base outputs) …")
            self.meta_model = LogisticRegression(C=1.0, max_iter=500)
            scaled = self.scaler.fit_transform(meta_X)
            self.meta_model.fit(scaled, y_val)
        else:
            print("\n[Ensemble] Only one base model — using direct predictions.")

        return self

    def _base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Return array of shape (n_samples, n_base_models)."""
        preds = []
        for name, model in self.base_models.items():
            try:
                p = _predict_proba(model, X)
                # Clip to [0,1]; LightGBM raw scores can exceed this
                p = np.clip(p, 0.0, 1.0)
                preds.append(p)
            except Exception as e:
                print(f"  [warn] {name} prediction failed: {e}")
        return np.column_stack(preds) if preds else np.empty((len(X), 0))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble probability for class 1."""
        meta_X = self._base_predictions(X)

        if meta_X.shape[1] == 0:
            raise RuntimeError("No base models produced predictions.")

        if self.meta_model is not None:
            scaled = self.scaler.transform(meta_X)
            return self.meta_model.predict_proba(scaled)[:, 1]

        # Weighted average fallback
        if self.weights:
            w = np.array([
                self.weights.get(n, 1.0)
                for n in list(self.base_models.keys())[:meta_X.shape[1]]
            ])
            w = w / w.sum()
        else:
            w = np.ones(meta_X.shape[1]) / meta_X.shape[1]
        return (meta_X * w).sum(axis=1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_DIR / "ensemble_v1.joblib"
        joblib.dump(self, path)
        print(f"[OK] Ensemble saved → {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "GolfEnsemble":
        path = path or MODEL_DIR / "ensemble_v1.joblib"
        return joblib.load(path)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(ensemble: GolfEnsemble, df: pd.DataFrame, label: str = "test") -> None:
    X = df[ensemble.feature_cols].fillna(0)
    y = _target(df)
    proba = ensemble.predict_proba(X)
    auc = roc_auc_score(y, proba)
    ll  = log_loss(y, proba)
    print(f"  {label:10s}  AUC {auc:.4f}  LogLoss {ll:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(predict_only: bool = False) -> None:
    df = _load_data()
    if "tournament_rank" not in df.columns:
        raise KeyError("'tournament_rank' required.")

    train_df, val_df, test_df = _time_split(df)

    if predict_only:
        print("Loading saved ensemble …")
        ens = GolfEnsemble.load()
    else:
        ens = GolfEnsemble()
        ens.fit(train_df, val_df)
        ens.save()

    print("\n[Ensemble] Evaluation:")
    evaluate(ens, val_df,  label="validation")
    evaluate(ens, test_df, label="test")

    # Compare against individual base-model AUC on test set
    feat_cols = ens.feature_cols
    X_test  = test_df[feat_cols].fillna(0)
    y_test  = _target(test_df).values
    print("\n[Base models on test set]:")
    for name, model in ens.base_models.items():
        try:
            p = _predict_proba(model, X_test)
            p = np.clip(p, 0, 1)
            print(f"  {name:10s}  AUC {roc_auc_score(y_test, p):.4f}")
        except Exception as e:
            print(f"  {name:10s}  failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacked ensemble model for golf predictions")
    parser.add_argument("--predict", action="store_true", help="Load and evaluate saved ensemble only")
    args = parser.parse_args()
    main(predict_only=args.predict)
