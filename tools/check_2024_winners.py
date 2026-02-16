import pandas as pd

df = pd.read_parquet('data_files/espn_pga_2024.parquet')

print(f"Total tournaments in 2024: {df.tournament.nunique()}")
print(f"\nChecking if Akshay Bhatia and Ryo Hisatsune always win...")

akshay_wins = 0
ryo_wins = 0
total_tourn = 0

for tourn in df.tournament.unique():
    tourn_df = df[df.tournament == tourn].copy()
    
    # Find who has the best (lowest) score
    best_score = tourn_df['total_score'].min()
    winners = tourn_df[tourn_df['total_score'] == best_score]['name'].tolist()
    
    if 'Akshay Bhatia' in winners:
        akshay_wins += 1
    if 'Ryo Hisatsune' in winners:
        ryo_wins += 1
    
    total_tourn += 1
    
    # Print first 5 tournaments
    if total_tourn <= 5:
        print(f"\n{tourn}:")
        print(f"  Best score: {best_score}")
        print(f"  Winner(s): {', '.join(winners)}")
        print("  Top 3:")
        top3 = tourn_df.nsmallest(3, 'total_score')[['name', 'total_score']]
        print(top3.to_string(index=False, header=False))

print(f"\n\nSummary across all {total_tourn} tournaments:")
print(f"  Akshay Bhatia wins: {akshay_wins}")
print(f"  Ryo Hisatsune wins: {ryo_wins}")
