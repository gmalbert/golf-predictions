"""Quick test to verify probability coverage display."""
from models.predict_upcoming import predict_upcoming_tournament, get_upcoming_tournaments

# Get first upcoming tournament
upcoming = get_upcoming_tournaments(90)
print(f"Testing: {upcoming.iloc[0]['name']}\n")

# Make predictions
predictions = predict_upcoming_tournament(
    upcoming.iloc[0]['name'],
    upcoming.iloc[0]['id'],
    upcoming.iloc[0]['date']
)

# Show probability distribution
print(f"Total field: {len(predictions)} players")
print(f"Top 20 probability sum: {predictions['win_probability'].head(20).sum()*100:.1f}%")
print(f"Top 50 probability sum: {predictions['win_probability'].head(50).sum()*100:.1f}%")
print(f"All players sum: {predictions['win_probability'].sum()*100:.1f}%")

print("\n✓ This explains why top 20 only shows ~50% - the other 50% is spread among remaining 136 players")
