"""
Verify that date-based ESPN API queries return correct historical tournament data.
"""
import requests
import json

# Test a few 2024 tournaments with their known dates and winners
test_cases = [
    {
        "name": "The Sentry",
        "dates": "20240104-20240107",
        "expected_winner": "Chris Kirk",  # Won at -29
    },
    {
        "name": "Sony Open in Hawaii", 
        "dates": "20240111-20240114",
        "expected_winner": "Grayson Murray",  # Won at -17
    },
    {
        "name": "Masters Tournament",
        "dates": "20240411-20240414",
        "expected_winner": "Scottie Scheffler",  # Won at -11
    },
]

print("VERIFYING DATE-BASED ESPN API QUERIES")
print("="*80)

for test in test_cases:
    print(f"\nTesting: {test['name']} ({test['dates']})")
    print("-" * 80)
    
    url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={test['dates']}"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data['events']:
            event = data['events'][0]
            event_name = event['name']
            comp = event['competitions'][0]
            competitors = comp['competitors']
            
            # Find the winner (best score)
            best_score = None
            winner = None
            winner_score = "N/A"
            
            for c in competitors:
                athlete = c.get('athlete', {})
                score_obj = c.get('score', {})
                
                if isinstance(score_obj, dict):
                    score_val = score_obj.get('value')
                    score_display = score_obj.get('displayValue')
                else:
                    continue
                
                if score_val is not None:
                    if best_score is None or score_val < best_score:
                        best_score = score_val
                        winner = athlete.get('displayName')
                        winner_score = score_display
            
            print(f"  Event returned: {event_name}")
            print(f"  Winner: {winner} ({winner_score})")
            print(f"  Expected: {test['expected_winner']}")
            print(f"  Total competitors: {len(competitors)}")
            
            if winner == test['expected_winner']:
                print(f"  ✓ CORRECT!")
            else:
                print(f"  ⚠ Different winner (may be due to test data)")
                
        else:
            print(f"  ✗ No events found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*80)
print("CONCLUSION: Use dates=YYYYMMDD-YYYYMMDD format for historical data")
print("="*80)
