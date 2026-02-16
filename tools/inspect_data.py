"""Quick data inspection script"""
import pandas as pd
from pathlib import Path

df = pd.read_parquet('data_files/espn_with_owgr_features.parquet')

print("="*70)
print("DATA STRUCTURE INSPECTION")
print("="*70)

print(f"\nTotal records: {len(df):,}")
print(f"Date range: {df['year'].min()}-{df['year'].max()}")

print("\n\nALL COLUMNS:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    nulls = df[col].isna().sum()
    pct_null = (nulls / len(df)) * 100
    print(f"{i:2d}. {col:30s} {str(dtype):15s} {pct_null:5.1f}% null")

print("\n\nSAMPLE RECORDS (Masters 2024):")
masters = df[(df['tournament'] == 'Masters Tournament') & (df['year'] == 2024)]
print(f"\nTotal players in Masters 2024: {len(masters)}")

cols_to_show = ['name', 'tournament_rank', 'prior_avg_score', 'owgr_rank_current', 
                'tournaments_last_365d', 'prior_count']
print(f"\nSample data:")
print(masters[cols_to_show].head(10).to_string())

print("\n\nWINNERS:")
winners = masters[masters['tournament_rank'] == 1.0]
print(winners[cols_to_show].to_string())

print("\n\nFEATURE CORRELATION WITH WIN:")
df['won'] = (df['tournament_rank'] == 1.0).astype(int)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
numeric_cols = [c for c in numeric_cols if c not in ['year', 'tournament_rank', 'won', 'player_id']]

correlations = df[numeric_cols + ['won']].corr()['won'].sort_values(ascending=False)
print("\nTop 15 features correlated with winning:")
print(correlations.head(15).to_string())
