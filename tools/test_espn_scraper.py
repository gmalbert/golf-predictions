"""Test ESPN scraper on a recent tournament"""
import sys
sys.path.append('scrapers')

from espn_golf import get_espn_leaderboard
import pandas as pd

# Test on 2024 Masters (hardcode event ID if known, or use API)
print("Testing ESPN scraper on 2024 Masters Tournament...")
print("="*70)

# Try to fetch 2024 Masters
event_id = "401580329"  # Example event ID
df = get_espn_leaderboard(event_id)

if not df.empty:
    print(f"\nTotal players found: {len(df)}")
    print(f"Unique names: {df['name'].nunique()}")
    
    print("\nTop 20 by position:")
    display = df[['name', 'position', 'total_score', 'total_strokes']].head(20)
    print(display.to_string(index=False))
    
    print("\nSample of all data columns:")
    print(df.head(5).to_dict('records'))
    
    # Check for duplicates
    dupes = df[df.duplicated(['name'], keep=False)]
    if not dupes.empty:
        print(f"\n⚠️  WARNING: {len(dupes)} duplicate player entries found!")
        print(dupes[['name', 'position', 'total_score']].to_string())
else:
    print("❌ No data returned from scraper!")
