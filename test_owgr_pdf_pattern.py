"""Test if OWGR PDFs follow a predictable URL pattern."""

import requests

# Test a few URL patterns
patterns = [
    "https://www.owgr.com/ranking/2026/week06.pdf",
    "https://www.owgr.com/ranking/2025/week52.pdf",
    "https://www.owgr.com/ranking/2025/week01.pdf",
]

for url in patterns:
    try:
        print(f"\n📡 Testing: {url}")
        resp = requests.get(url, timeout=10, allow_redirects=True)
        print(f"   Status: {resp.status_code}")
        print(f"   Content-Type: {resp.headers.get('Content-Type')}")
        if resp.status_code == 200:
            print(f"   ✓ PDF found! Size: {len(resp.content):,} bytes")
        else:
            print(f"   ✗ Not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
