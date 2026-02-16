"""Inspect ESPN API response structure to fix position extraction"""
import sys
sys.path.append('scrapers')

from shared_utils import polite_get
import json

# Fetch a tournament
event_id = "401580329"  # 2024 Masters
url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?event={event_id}"

print("Fetching ESPN API response...")
resp = polite_get(url, use_cache=True)
data = resp.json()

# Navigate to competitors
event = data.get("events", [{}])[0]
competition = event.get("competitions", [{}])[0]
competitors = competition.get("competitors", [])

print(f"\nFound {len(competitors)} competitors")
print("\nInspecting first 3 competitors to understand structure:\n")

for i, c in enumerate(competitors[:3]):
    print(f"{'='*70}")
    print(f"COMPETITOR {i+1}: {c.get('athlete', {}).get('displayName')}")
    print(f"{'='*70}")
    
    print(f"\nFull competitor dict keys: {list(c.keys())}")
    
    print(f"\n  athlete: {c.get('athlete', {})}")
    print(f"\n  status: {c.get('status', {})}")
    
    status = c.get('status', {})
    print(f"\n  status keys: {list(status.keys())}")
    print(f"  status.type: {status.get('type')}")
    print(f"  status.position: {status.get('position')}")
    
    print(f"\n  score: {c.get('score')}")
    print(f"\n  sortOrder: {c.get('sortOrder')}")
    print(f"\n  linescores: {c.get('linescores')}")
    
    print("\n")

# Also check if there's ranking info elsewhere
print(f"\n{'='*70}")
print("CHECKING EVENT-LEVEL DATA")
print(f"{'='*70}")
print(f"Event keys: {list(event.keys())}")
print(f"Competition keys: {list(competition.keys())}")
