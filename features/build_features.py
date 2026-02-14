"""
Build features from the consolidated ESPN PGA dataset.

Saves a player-tournament feature table suitable for model training.

Key features (no leakage):
- numeric_total_score: parsed from `total_score`
- tournament_rank: computed per tournament (1 = winner)
- prior_count: number of prior events for player
- prior_avg_score / prior_std_score
- prior_avg_score_5 / prior_avg_score_10
- prior_top10_rate_5 / prior_top10_rate_10
- days_since_last_event
- tournaments_last_365d
- season_to_date_avg_score
- course_history_avg_score (at same tournament/course)
- last_event_score / last_event_rank

Usage:
    python features/build_features.py --in data_files/espn_pga_2018_2025.parquet \
        --out data_files/espn_player_tournament_features.parquet
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def parse_score(s):
    if pd.isna(s):
        return np.nan
    try:
        s = str(s).strip()
        if s == "E":
            return 0
        # remove plus sign
        if s.startswith('+'):
            return int(s[1:])
        return int(s)
    except Exception:
        return np.nan


def safe_rank(series):
    # Lower score is better (more negative). Rank 1 = best.
    # series contains numeric_total_score (smaller = better)
    # NaNs should receive NaN rank.
    s = series.copy()
    mask = s.notna()
    ranks = pd.Series(index=s.index, dtype=float)
    if mask.any():
        ranks.loc[mask] = s[mask].rank(method='min', ascending=True)
    return ranks


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure date is datetime and sort chronologically
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(['date', 'tournament_id', 'player_id']).reset_index(drop=True)

    # Numeric total score (relative to par). Lower is better.
    df['numeric_total_score'] = df['total_score'].apply(parse_score).astype('float')

    # Tournament rank computed from numeric_total_score
    df['tournament_rank'] = df.groupby('tournament_id')['numeric_total_score'].transform(safe_rank)

    # Replace impossible ranks (NaN) with NaN float
    df['tournament_rank'] = df['tournament_rank'].astype('float')

    # Per-player chronological features (cumulative / shifted so current row only sees past)
    # prior_count: number of tournaments played before this event
    df['prior_count'] = df.groupby('player_id').cumcount()

    # prior cumulative sums and means (shifted)
    grp = df.groupby('player_id')

    # cumulative sum / count of numeric_total_score (include NaN handling) - use transform for alignment
    df['cum_sum_score'] = grp['numeric_total_score'].transform(lambda s: s.cumsum().shift(1))
    df['cum_count_score'] = grp['numeric_total_score'].transform(lambda s: s.notna().cumsum().shift(1))
    df['prior_avg_score'] = df['cum_sum_score'] / df['cum_count_score']
    df.loc[df['cum_count_score'] == 0, 'prior_avg_score'] = np.nan

    # cumulative std (use expanding().std() then shift)
    df['prior_std_score'] = grp['numeric_total_score'].transform(lambda s: s.expanding().std().shift(1))

    # last_event_score and last_event_rank
    df['last_event_score'] = grp['numeric_total_score'].shift(1)
    df['last_event_rank'] = grp['tournament_rank'].shift(1)

    # Rolling features for last N events (5, 10) — use transform for proper alignment
    for window in (5, 10):
        df[f'prior_avg_score_{window}'] = grp['numeric_total_score'].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        df[f'prior_std_score_{window}'] = grp['numeric_total_score'].transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())

        # top-10 rate in last `window` events
        def _top10_rate(s):
            return s.shift(1).rolling(window, min_periods=1).apply(lambda x: np.nanmean((x<=10).astype(float)), raw=False)

        df[f'prior_top10_rate_{window}'] = grp['tournament_rank'].transform(_top10_rate)

    # Days since last event
    df['days_since_last_event'] = (df['date'] - grp['date'].shift(1)).dt.total_seconds() / 86400.0

    # Events played in last 365 days — helper and transform to preserve index alignment
    def events_in_window(dates, days=365):
        # dates is a pd.Series of datetimes (sorted for each player)
        out = pd.Series(index=dates.index, dtype=int)
        arr = dates.values
        # For each position i, count prior dates > cutoff
        for i in range(len(arr)):
            if i == 0:
                out.iloc[i] = 0
                continue
            cutoff = arr[i] - np.timedelta64(days, 'D')
            cnt = (arr[:i] > cutoff).sum()
            out.iloc[i] = int(cnt)
        return out

    df['tournaments_last_365d'] = grp['date'].transform(lambda x: events_in_window(x, days=365))

    # season-to-date average (within same year) — exclude current
    df['season_to_date_avg_score'] = df.groupby(['player_id', 'year'])['numeric_total_score'].transform(lambda s: s.shift(1).expanding().mean())

    # course / tournament history avg (exclude current)
    df['course_history_avg_score'] = df.groupby(['player_id', 'tournament'])['numeric_total_score'].transform(lambda s: s.shift(1).expanding().mean())

    # career best rank so far (min rank excluding current)
    df['career_best_rank'] = grp['tournament_rank'].transform(lambda s: s.shift(1).cummin())

    # played_recent flag (played within last 30 days)
    df['played_last_30d'] = (df['days_since_last_event'] <= 30).astype('float')

    # Fill infs and keep relevant columns
    feature_cols = [
        'player_id', 'name', 'tournament', 'tournament_id', 'date', 'year', 'country',
        'numeric_total_score', 'tournament_rank',
        'prior_count', 'prior_avg_score', 'prior_std_score', 'last_event_score', 'last_event_rank',
        'prior_avg_score_5', 'prior_avg_score_10', 'prior_std_score_5', 'prior_std_score_10',
        'prior_top10_rate_5', 'prior_top10_rate_10', 'days_since_last_event', 'tournaments_last_365d',
        'season_to_date_avg_score', 'course_history_avg_score', 'career_best_rank', 'played_last_30d'
    ]

    # Ensure columns exist (some may be missing if dataset had insufficient data)
    existing = [c for c in feature_cols if c in df.columns]
    features = df[existing].copy()

    # Clean types
    features['tournaments_last_365d'] = features['tournaments_last_365d'].fillna(0).astype(int)

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build player-tournament features')
    parser.add_argument('--in', dest='infile', type=str, default='data_files/espn_pga_2018_2025.parquet')
    parser.add_argument('--out', dest='outfile', type=str, default='data_files/espn_player_tournament_features.parquet')

    args = parser.parse_args()
    infile = Path(args.infile)
    outfile = Path(args.outfile)

    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    print(f"Loading: {infile}")
    df = pd.read_parquet(infile)
    print(f"Rows loaded: {len(df):,}")

    feats = build_features(df)
    print(f"Features built: {feats.shape}")

    feats.to_parquet(outfile, index=False)
    print(f"💾 Saved features to: {outfile}")
