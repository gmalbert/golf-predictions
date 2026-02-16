"""
Debug score structure from ESPN API date-based queries.
"""
import requests
import json

url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates=20240104-20240107"

print("Fetching The Sentry 2024...")
resp = requests.get(url, timeout=10)
data = resp.json()

event = data['events'][0]
comp = event['competitions'][0]
competitors = comp['competitors']

print(f"\nTotal competitors: {len(competitors)}")
print("\nFirst 3 competitors - Full structure:")

for i, c in enumerate(competitors[:3]):
    print(f"\n{'='*80}")
    print(f"Competitor {i+1}:")
    print(f"  Athlete: {c.get('athlete', {}).get('displayName')}")
    print(f"  Score type: {type(c.get('score'))}")
    print(f"  Score: {c.get('score')}")
    print(f"  SortOrder: {c.get('sortOrder')}")
    print(f"  Status: {c.get('status')}")

# Try to find the actual winner
print(f"\n{'='*80}")
print("All scores (first 10):")
for i, c in enumerate(competitors[:10]):
    name = c.get('athlete', {}).get('displayName', 'Unknown')
    score = c.get('score')
    sortOrder = c.get('sortOrder')
    print(f"  {i+1}. {name:30} Score: {score:10} SortOrder: {sortOrder}")
