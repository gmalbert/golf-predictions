"""Test Next.js data endpoint for OWGR."""

import requests
from shared_utils import polite_get, get_random_headers

build_id = "Q-ts6kTnv5CsZNlfPOQ8A"
slug = "ranking"  # From the URL path

# Next.js data endpoint pattern
url = f"https://www.owgr.com/_next/data/{build_id}/{slug}.json"

print(f"Trying Next.js data endpoint:")
print(f"  URL: {url}\n")

try:
    resp = requests.get(url, headers=get_random_headers(), timeout=15)
    print(f"Status: {resp.status_code}")
    
    if resp.status_code == 200:
        print(f"✓ Success!")
        
        try:
            data = resp.json()
            print(f"\nJSON keys: {list(data.keys())}")
            
            if "pageProps" in data:
                print(f"\npageProps keys: {list(data['pageProps'].keys())}")
                
                # Look for ranking data
                def find_data(obj, depth=0, max_depth=5):
                    if depth > max_depth:
                        return
                    
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if "rank" in key.lower() or "player" in key.lower():
                                print(f"\n🎯 Found '{key}' at depth {depth}")
                                print(f"   Type: {type(value)}")
                                if isinstance(value, list):
                                    print(f"   Length: {len(value)}")
                                    if value:
                                        print(f"   First item: {value[0]}")
                            
                            if isinstance(value, (dict, list)):
                                find_data(value, depth + 1, max_depth)
                    
                    elif isinstance(obj, list) and obj:
                        find_data(obj[0], depth, max_depth)
                
                print("\n" + "="*60)
                print("Searching for ranking data...")
                print("="*60)
                find_data(data)
            
            # Save for inspection
            import json
            from pathlib import Path
            Path("data_files/owgr_nextjs_data.json").write_text(json.dumps(data, indent=2))
            print(f"\n💾 Saved to: data_files/owgr_nextjs_data.json")
        
        except Exception as e:
            print(f"Error parsing: {e}")
            print(f"Response preview: {resp.text[:500]}")
    
    else:
        print(f"❌ Failed with status {resp.status_code}")
        print(f"Response: {resp.text[:300]}")

except Exception as e:
    print(f"❌ Error: {e}")
