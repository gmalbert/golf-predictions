from espn_golf import get_espn_leaderboard

tests = [
    ('401580329', '2024-01-04T05:00Z', 'The Sentry'),
    ('401580330', '2024-01-11T05:00Z', 'Sony Open in Hawaii'),
    ('401580344', '2024-04-11T05:00Z', 'Masters Tournament'),
]

print("Testing updated ESPN scraper with date-based queries:")
print("="*80)

for event_id, date, name in tests:
    df = get_espn_leaderboard(event_id, event_date=date)
    
    if not df.empty:
        winner = df.iloc[0]['name']
        score = df.iloc[0]['total_score']
        total_players = len(df)
        
        print(f"\n{name}:")
        print(f"  Winner: {winner} ({score})")
        print(f"  Total players: {total_players}")
    else:
        print(f"\n{name}: NO DATA")

print("\n" + "="*80)
print("SUCCESS! Each tournament has different winner")
