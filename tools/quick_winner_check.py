import pandas as pd

df = pd.read_parquet('data_files/espn_player_tournament_features.parquet')

winners = df[df['tournament_rank'] == 1.0]

print(f"Total tournaments: {df.tournament.nunique()}")
print(f"Total winner records: {len(winners)}")
print(f"\nTop 20 players by wins:")
print(winners.name.value_counts().head(20))

print(f"\n\nUnique winners: {winners.name.nunique()}")

# Check if we have the old corruption (Akshay/Ryo winning everything)
akshay_wins = len(winners[winners.name.str.contains('Akshay', na=False)])
ryo_wins = len(winners[winners.name.str.contains('Ryo', na=False)])

print(f"\nAkshay Bhatia wins: {akshay_wins}")
print(f"Ryo Hisatsune wins: {ryo_wins}")

if akshay_wins > 50 or ryo_wins > 50:
    print("\n❌ DATA STILL CORRUPTED!")
else:
    print("\n✅ DATA IS CLEAN! No more corruption!")
