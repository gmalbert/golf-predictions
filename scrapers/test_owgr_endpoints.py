"""Try different OWGR API endpoints."""

import requests
from shared_utils import polite_get, get_random_headers

# Common API endpoint patterns to try
endpoints = [
    "https://www.owgr.com/api/v1/ranking",
    "https://www.owgr.com/api/v1/rankings",
    "https://www.owgr.com/api/rankings",
    "https://www.owgr.com/api/players/ranking",
    "https://www.owgr.com/_next/data/ranking",
    "https://data.owgr.com/ranking",
    "https://api.owgr.com/v1/ranking",
]

print("Testing OWGR API endpoints...\n")

for endpoint in endpoints:
    print(f"Trying: {endpoint}")
    try:
        resp = requests.get(endpoint, headers=get_random_headers(), timeout=10)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            print(f"  Content-Type: {content_type}")
            
            if "json" in content_type:
                try:
                    data = resp.json()
                    print(f"  ✓ JSON response!")
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else 'array'}")
                    print(f"  Preview: {str(data)[:200]}")
                except:
                    print(f"  ⚠️  Not valid JSON")
            else:
                print(f"  Preview: {resp.text[:200]}")
        
        elif resp.status_code in [301, 302, 307, 308]:
            print(f"  Redirect to: {resp.headers.get('Location')}")
    
    except requests.exceptions.RequestException as e:
        print(f"  Error: {e}")
    
    print()
