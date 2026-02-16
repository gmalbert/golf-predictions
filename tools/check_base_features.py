import pandas as pd

# Check the feature file BEFORE OWGR features are added
df = pd.read_parquet('data_files/espn_player_tournament_features.parquet')

print(f"ESPN features file shape: {df.shape}")
print(f"\nplayer_id stats:")
print(f"  Null: {df.player_id.isna().sum()} / {len(df)} ({df.player_id.isna().sum()/len(df)*100:.1f}%)")
print(f"  Unique: {df.player_id.nunique()}")

# Check a specific player's features
print(f"\nScottie Scheffler 2024 sample:")
scottie = df[(df.name.str.contains('Scottie', na=False)) & (df.year == 2024)]
if not scottie.empty:
    cols = ['name', 'player_id', 'tournament', 'prior_avg_score', 'last_event_rank', 'career_best_rank', 'prior_count']
    print(scottie[cols].head(3).to_string(index=False))
    
    # Check feature completeness for Scottie
    print(f"\nFeature stats for Scottie in 2024:")
    for col in ['prior_avg_score', 'last_event_rank', 'career_best_rank', 'prior_count']:
        valid = scottie[col].notna().sum()
        total = len(scottie)
        print(f"  {col}: {valid}/{total} valid ({valid/total*100:.1f}%)")
else:
    print("  No Scottie Scheffler found")

# Overall feature stats
print(f"\nOverall feature completeness:")
feature_cols = ['prior_avg_score', 'last_event_rank', 'career_best_rank', 'prior_count']
for col in feature_cols:
    pct_valid = (df[col].notna().sum() / len(df)) * 100
    print(f"  {col}: {pct_valid:.1f}% valid")
