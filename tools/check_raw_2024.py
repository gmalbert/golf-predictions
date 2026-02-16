import pandas as pd
import sys
sys.path.insert(0, 'c:\\Users\\gmalb\\Downloads\\golf-predictions')
from features.build_features import parse_score

df = pd.read_parquet('data_files/espn_pga_2024.parquet')

# Convert to numeric properly
df['numeric_score'] = df['total_score'].apply(parse_score)

print(f"Total 2024 tournaments: {df.tournament.nunique()}\n")

akshay_best = 0
ryo_best = 0

for tourn in df.tournament.unique()[:10]:  # Check first 10
    tourn_df = df[df.tournament == tourn].copy()
    
    # Find minimum numeric score (best)
    best_score = tourn_df['numeric_score'].min()
    winners = tourn_df[tourn_df['numeric_score'] == best_score]
    
    print(f"{tourn}:")
    print(f"  Best numeric score: {best_score}")
    print(f"  Winners: {', '.join(winners['name'].tolist())}")
    
    if 'Akshay Bhatia' in winners['name'].values:
        akshay_best += 1
    if 'Ryo Hisatsune' in winners['name'].values:
        ryo_best += 1
    
    # Show all Akshay/Ryo scores
    for name in ['Akshay Bhatia', 'Ryo Hisatsune']:
        player_row = tourn_df[tourn_df['name'] == name]
        if not player_row.empty:
            score = player_row['numeric_score'].iloc[0]
            print(f"    {name}: {score}")
    print()

print(f"\nIn first 10 tournaments:")
print(f"  Akshay was best: {akshay_best} times")
print(f"  Ryo was best: {ryo_best} times")
