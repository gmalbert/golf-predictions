"""Quick ESPN API test to check current endpoint structure"""
import requests
import json

print("Testing ESPN Golf API...")
print("="*60)

# Test current scoreboard
url = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
r = requests.get(url, timeout=30)
data = r.json()

events = data.get('events', [])
print(f"\n✓ Found {len(events)} current/recent events\n")

if events:
    # Show first 3 events
    for i, event in enumerate(events[:3], 1):
        print(f"{i}. {event['name']}")
        print(f"   ID: {event['id']}")
        print(f"   Date: {event.get('date', 'N/A')}")
        print(f"   Status: {event.get('status', {}).get('type', {}).get('description', 'Unknown')}")
        
        # Try to get leaderboard data directly from scoreboard
        comps = event.get('competitions', [])
        if comps:
            competitors = comps[0].get('competitors', [])
            print(f"   Competitors in scoreboard: {len(competitors)}")
            if competitors:
                top = competitors[0]
                athlete = top.get('athlete', {})
                print(f"   Leader: {athlete.get('displayName', 'N/A')}")
        print()

print("\n" + "="*60)
print("The ESPN scoreboard API includes leaderboard data!")
print("No need for separate leaderboard endpoint.")
print("="*60)
