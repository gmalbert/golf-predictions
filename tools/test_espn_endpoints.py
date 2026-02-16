"""
Test different ESPN API endpoints to find one that returns historical tournament data.
"""
import requests
import json
from pathlib import Path

def check_endpoint(url, description):
    """Check an ESPN API endpoint and return basic info (script-style helper)."""
    print(f"\n{'='*80}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"{'='*80}")
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Try to extract event info
        if 'events' in data and data['events']:
            event = data['events'][0]
            print(f"✓ Success!")
            print(f"  Event Name: {event.get('name', 'N/A')}")
            print(f"  Event ID: {event.get('id', 'N/A')}")
            print(f"  Date: {event.get('date', 'N/A')}")
            
            # Check for competition data
            if 'competitions' in event and event['competitions']:
                comp = event['competitions'][0]
                competitors = comp.get('competitors', [])
                print(f"  Competitors: {len(competitors)}")
                
                if competitors:
                    print(f"\n  Sample competitor:")
                    c = competitors[0]
                    athlete = c.get('athlete', {})
                    score = c.get('score', {})
                    if isinstance(score, dict):
                        score_display = score.get('displayValue', 'N/A')
                    else:
                        score_display = str(score)
                    print(f"    Name: {athlete.get('displayName', 'N/A')}")
                    print(f"    Score: {score_display}")
            
            return True
        else:
            print(f"✗ No events found in response")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


# Test different endpoint patterns
print("TESTING ESPN GOLF API ENDPOINTS FOR HISTORICAL DATA")
print("="*80)

# Pattern 1: Scoreboard with event parameter (current broken approach)
check_endpoint(
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?event=401580329",
    "Scoreboard with event= parameter (The Sentry 2024)"
)

# Pattern 2: Direct event endpoint
check_endpoint(
    "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/401580329",
    "Direct event endpoint"
)

# Pattern 3: Event summary
check_endpoint(
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/summary?event=401580329",
    "Event summary endpoint"
)

# Pattern 4: Leaderboard endpoint
check_endpoint(
    "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/events/401580329/competitions/401580329/competitors",
    "Direct competitors endpoint"
)

# Pattern 5: Try year-based schedule
check_endpoint(
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates=2024&limit=100",
    "Schedule with dates=2024"
)

# Pattern 6: Try specific date range
check_endpoint(
    "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates=20240104-20240107",
    "Schedule with specific date range (The Sentry dates)"
)

# Pattern 7: Try calendar endpoint
check_endpoint(
    "https://sports.core.api.espn.com/v2/sports/golf/leagues/pga/seasons/2024/types/1/events",
    "Calendar/events list for 2024"
)

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
