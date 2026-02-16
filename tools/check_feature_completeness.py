import pandas as pd

df = pd.read_parquet('data_files/espn_with_owgr_features.parquet')

print(f"Total rows: {len(df)}")
print(f"Years: {df.year.min()}-{df.year.max()}")

# Check 2024 data specifically
df_2024 = df[df.year == 2024].copy()
print(f"\n2024 data: {len(df_2024)} rows")

# Check feature completeness
feature_cols = [
    'prior_avg_score', 'last_event_rank', 'career_best_rank',
    'owgr_rank_current', 'prior_count', 'tournaments_last_365d'
]

print(f"\nFeature completeness in 2024:")
for col in feature_cols:
    pct_valid = (df_2024[col].notna().sum() / len(df_2024)) * 100
    print(f"  {col}: {pct_valid:.1f}% valid")

# Sample from 2024 Masters
print(f"\n2024 Masters sample (first 5 players):")
masters = df_2024[df_2024.tournament.str.contains('Masters', na=False)]
if not masters.empty:
    sample_cols = ['name', 'prior_avg_score', 'last_event_rank', 'career_best_rank', 'owgr_rank_current']
    print(masters[sample_cols].head().to_string())
else:
    print("  No Masters data found in 2024")

# Check overall feature completeness
print(f"\nOverall feature completeness (all years):")
for col in feature_cols:
    pct_valid = (df[col].notna().sum() / len(df)) * 100
    print(f"  {col}: {pct_valid:.1f}% valid")
