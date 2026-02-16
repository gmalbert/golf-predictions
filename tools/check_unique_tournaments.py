import pandas as pd
import sys
sys.path.insert(0, 'c:\\Users\\gmalb\\Downloads\\golf-predictions')
from features.build_features import parse_score

df = pd.read_parquet('data_files/espn_pga_2024.parquet')
df['numeric_score'] = df['total_score'].apply(parse_score)

print(f"Total rows: {len(df)}")
print(f"Unique tournaments: {df.tournament.nunique()}")
print(f"\nFirst 5 tournaments:\n")

for i, t in enumerate(df.tournament.unique()[:5]):
    t_df = df[df.tournament == t].copy()
    best_score = t_df['numeric_score'].min()
    winner = t_df[t_df['numeric_score'] == best_score].iloc[0]['name']
    
    print(f"{i+1}. {t}")
    print(f"   Winner: {winner} ({best_score})")
    print(f"   Total players: {len(t_df)}")
