"""Check a comp leted tournament for position data"""
import sys
sys.path.append('scrapers')

from shared_utils import polite_get
import json

# Try a completed 2023 tournament
event_id = "401465515"  # 2023 Masters (should be complete)
url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?event={event_id}"

print("Fetching completed 2023 Masters Tournament...")
resp = polite_get(url, use_cache=False)
data = resp.json()

event = data.get("events", [{}])[0]
competition = event.get("competitions", [{}])[0]
competitors = competition.get("competitors", [])

print(f"Found {len(competitors)} competitors")
print(f"Event status: {event.get('status', {}).get('type', {}).get('name')}")

print("\nFirst 5 competitors (should have position data):\n")

for i, c in enumerate(competitors[:5]):
    name = c.get('athlete', {}).get('displayName')
    score = c.get('score')
    sortOrder = c.get('sortOrder')
    status = c.get('status', {})
    
    print(f"{i+1}. {name}")
    print(f"   Score: {score}")
    print(f"   sortOrder: {sortOrder}")
    print(f"   status: {status}")
    print(f"   status.position: {status.get('position')}")
    print()
