"""Diagnose the data quality issue"""
import pandas as pd
import numpy as np

# Load raw ESPN data
df = pd.read_parquet('data_files/espn_pga_2018_2025.parquet')

print("="*70)
print("DATA QUALITY DIAGNOSIS")
print("="*70)

# Check 2018 Masters
masters_2018 = df[df['tournament'] == '2018 Masters Tournament'].copy()

print(f"\n2018 Masters Tournament:")
print(f"  Total players: {len(masters_2018)}")
print(f"  Unique scores: {masters_2018['total_score'].nunique()}")

# Parse scores
def parse_score(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s == "E":
        return 0
    if s.startswith('+'):
        return int(s[1:])  # +1 becomes 1
    return int(s)  # -10 becomes -10

masters_2018['numeric_score'] = masters_2018['total_score'].apply(parse_score)

# Sort by score (lower is better)
masters_sorted = masters_2018.sort_values('numeric_score')

print(f"\nTop 10 finishers (BEST scores):")
print(masters_sorted[['name', 'total_score', 'numeric_score']].head(10).to_string(index=False))

print(f"\nBottom 10 finishers (WORST scores):")
print(masters_sorted[['name', 'total_score', 'numeric_score']].tail(10).to_string(index=False))

# Check if there are players with -15
best_score = masters_2018['numeric_score'].min()
best_players = masters_2018[masters_2018['numeric_score'] == best_score]

print(f"\n\nBest score: {best_score}")
print(f"Players with best score: {len(best_players)}")
print(best_players[['name', 'total_score']].to_string(index=False))

# Check across all tournaments
print(f"\n\n{'='*70}")
print("OVERALL DATASET CHECK")
print(f"{'='*70}")

df['numeric_score'] = df['total_score'].apply(parse_score)

# Find the  absolute best score ever recorded
overall_best = df['numeric_score'].min()
best_ever = df[df['numeric_score'] == overall_best]

print(f"\nBest score ever in dataset: {overall_best}")
print(f"Number of times this score appears: {len(best_ever)}")
print(f"Players who achieved it:")
print(best_ever.groupby('name')['tournament'].count().sort_values(ascending=False).head(10).to_string())

# Check how many tournaments each player "won" (lowest score)
print(f"\n\nChecking who has the most 'wins' (lowest scores in tournaments):")

def get_winners(group):
    min_score = group['numeric_score'].min()
    winners = group[group['numeric_score'] == min_score]
    return winners['name'].tolist()

tournament_winners = df.groupby(['tournament','year']).apply(get_winners).reset_index()
tournament_winners.columns = ['tournament', 'year', 'winners']

# Explode the winners list
all_winners = []
for _, row in tournament_winners.iterrows():
    for winner in row['winners']:
        all_winners.append(winner)

from collections import Counter
winner_counts = Counter(all_winners)

print("\nTop 10 'winners' (most tournaments with lowest score):")
for name, count in winner_counts.most_common(10):
    print(f"  {name}: {count} wins")
