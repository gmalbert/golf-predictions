"""
Walk-Forward Backtesting Engine

Simulates betting on every tournament from a given start year using a
strict walk-forward protocol: the model is retrained on all data available
before each tournament, then evaluated on that tournament's actual results.

Usage
-----
    python evaluation/backtester.py                       # defaults
    python evaluation/backtester.py --start-year 2022 --kelly 0.25
    python evaluation/backtester.py --no-betting          # metrics only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data_files"


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_feature_parquet() -> pd.DataFrame:
    for p in [
        DATA_DIR / "espn_with_extended_features.parquet",
        DATA_DIR / "espn_with_owgr_features.parquet",
        DATA_DIR / "espn_player_tournament_features.parquet",
    ]:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError("No feature parquet found in data_files/")


def _load_feature_cols() -> list[str]:
    from models.evaluate_model_stats import load_model_and_features
    _, cols = load_model_and_features()
    return cols


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest(
    start_year: int = 2022,
    kelly_fraction: float = 0.25,
    initial_bankroll: float = 1_000.0,
    simulate_betting: bool = True,
    min_train_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest.

    For each tournament ≥ start_year, in chronological order:
      1. Train on all data whose date precedes the tournament date.
      2. Predict win probabilities for the tournament field.
      3. Evaluate top-N accuracy and AUC.
      4. Optionally simulate a fractional-Kelly bet on the model's top pick.

    Returns
    -------
    results_df : Per-tournament metrics DataFrame.
    summary_df : Aggregate statistics DataFrame.
    """
    import xgboost as xgb

    try:
        feature_cols = _load_feature_cols()
    except Exception as e:
        print(f"[WARN] Could not load model features ({e}); defaulting to common features.")
        feature_cols = [
            "prior_avg_score", "prior_avg_score_5", "prior_avg_score_10",
            "prior_top10_rate_5", "prior_top10_rate_10", "days_since_last_event",
            "owgr_rank_current", "owgr_points_current", "is_major", "purse_tier",
        ]

    df = _load_feature_parquet()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    df["won"] = (df["tournament_rank"] == 1).astype(int)

    tournaments = (
        df[df["year"] >= start_year]
        .groupby(["tournament", "year"])
        .agg(date=("date", "min"))
        .reset_index()
        .sort_values("date")
    )

    bankroll = initial_bankroll
    results = []

    for _, row in tournaments.iterrows():
        tourney, year, t_date = row["tournament"], row["year"], row["date"]

        train = df[df["date"] < t_date].copy()
        test  = df[(df["tournament"] == tourney) & (df["year"] == year)].copy()

        if len(train) < min_train_rows or test.empty:
            continue
        if test["won"].sum() == 0:
            continue   # no recorded winner – skip

        # Prepare features
        X_tr = train.reindex(columns=feature_cols).fillna(0)
        y_tr = train["won"]
        X_te = test.reindex(columns=feature_cols).fillna(0)
        y_te = test["won"].values

        # Fit a fresh XGBoost on all prior data
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()),
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=1, verbosity=0,
        )
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_te)[:, 1]
        test  = test.copy()
        test["pred_prob"] = probs

        # Top-N accuracy
        top_n_hits = {}
        for n in [1, 3, 5, 10, 20]:
            top_idx = test.nlargest(n, "pred_prob").index
            hit = int(test.loc[test.index.isin(top_idx), "won"].sum() > 0)
            top_n_hits[f"top{n}_hit"] = hit

        # AUC (only if field has both classes)
        auc = float("nan")
        try:
            auc = roc_auc_score(y_te, probs)
        except Exception:
            pass

        # Fractional Kelly betting simulation
        bet_amount = bet_profit = 0.0
        estimated_odds = 21.0   # ~+2000 American odds (typical winner payout)
        if simulate_betting:
            from betting.bankroll import BankrollManager
            mgr = BankrollManager(bankroll, kelly_fraction)
            best = test.loc[test["pred_prob"].idxmax()]
            bet_amount = mgr.calculate_bet_size(float(best["pred_prob"]), estimated_odds)
            if bet_amount > 0:
                won_bet = bool(best["won"] == 1)
                rec = mgr.place_bet(
                    str(best["name"]), bet_amount, estimated_odds, won_bet, tourney
                )
                bet_profit = rec["profit"]
                bankroll = mgr.bankroll

        results.append({
            "tournament": tourney,
            "year":       int(year),
            "date":       str(t_date.date()),
            "field_size": len(test),
            "auc":        round(auc, 4),
            **top_n_hits,
            "bet_amount":  round(bet_amount, 2),
            "bet_profit":  round(bet_profit, 2),
            "bankroll":    round(bankroll, 2),
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("[WARN] No tournaments matched the criteria.")
        return results_df, pd.DataFrame()

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results_df)
    summary = {
        "tournaments_evaluated": n,
        "mean_auc":     round(results_df["auc"].mean(), 4),
        "top1_accuracy":  f"{results_df['top1_hit'].mean():.1%}",
        "top3_accuracy":  f"{results_df['top3_hit'].mean():.1%}",
        "top5_accuracy":  f"{results_df['top5_hit'].mean():.1%}",
        "top10_accuracy": f"{results_df['top10_hit'].mean():.1%}",
        "top20_accuracy": f"{results_df['top20_hit'].mean():.1%}",
    }
    if simulate_betting:
        total_wagered = results_df["bet_amount"].sum()
        total_profit  = results_df["bet_profit"].sum()
        summary.update({
            "total_wagered":         f"${total_wagered:,.2f}",
            "total_profit":          f"${total_profit:+,.2f}",
            "betting_roi":           f"{total_profit / total_wagered * 100:+.1f}%" if total_wagered else "N/A",
            "final_bankroll":        f"${results_df['bankroll'].iloc[-1]:,.2f}",
        })

    summary_df = pd.DataFrame([summary])
    return results_df, summary_df


def _print_results(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    print("\n── Walk-Forward Backtest Results ──────────────────────────────")
    print(results_df[[
        "tournament", "year", "field_size", "auc",
        "top1_hit", "top5_hit", "top10_hit", "bet_profit", "bankroll",
    ]].to_string(index=False))

    print("\n── Summary ────────────────────────────────────────────────────")
    for k, v in summary_df.iloc[0].items():
        print(f"  {k:<28} {v}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--start-year",    type=int,   default=2022)
    parser.add_argument("--kelly",         type=float, default=0.25)
    parser.add_argument("--bankroll",      type=float, default=1000.0)
    parser.add_argument("--no-betting",    action="store_true")
    parser.add_argument("--min-train",     type=int,   default=500)
    args = parser.parse_args()

    results, summary = run_backtest(
        start_year=args.start_year,
        kelly_fraction=args.kelly,
        initial_bankroll=args.bankroll,
        simulate_betting=not args.no_betting,
        min_train_rows=args.min_train,
    )
    _print_results(results, summary)

    # Save to disk
    if not results.empty:
        out = DATA_DIR / "backtest_results.csv"
        results.to_csv(out, index=False)
        print(f"[OK] Results saved → {out}")
