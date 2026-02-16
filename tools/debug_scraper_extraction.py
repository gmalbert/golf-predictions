import sys
sys.path.insert(0, 'c:\\Users\\gmalb\\Downloads\\golf-predictions')

from scrapers.espn_golf import get_espn_leaderboard

# Test on Masters (event 401580344)
print("Testing get_espn_leaderboard on 2024 Masters...")
df = get_espn_leaderboard("401580344")

print(f"\nDataFrame shape: {df.shape}")
print("\nFirst 10 players extracted:")
print(df[['name', 'total_score']].head(10))

print("\nAkshay Bhatia's score:")
akshay = df[df['name'].str.contains('Akshay', na=False)]
print(akshay[['name', 'total_score']])

print("\nRyo Hisatsune's score:")
ryo = df[df['name'].str.contains('Ryo', na=False)]
print(ryo[['name', 'total_score']])
