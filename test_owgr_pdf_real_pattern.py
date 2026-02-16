"""Test OWGR PDF pattern with actual filename format."""

import requests

# Test the real pattern: owgr{week}f{year}v2.pdf
base_urls = [
    "https://www.owgr.com/",
    "https://www.owgr.com/pdfs/",
    "https://www.owgr.com/ranking/",
    "https://www.owgr.com/archive/",
]

filenames = [
    "owgr01f2024v2.pdf",  # Week 1, 2024
    "owgr52f2025v2.pdf",  # Week 52, 2025
    "owgr06f2026v2.pdf",  # Week 6, 2026
]

for base in base_urls:
    for filename in filenames:
        url = base + filename
        try:
            print(f"\n📡 {url}")
            resp = requests.get(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200 and "pdf" in resp.headers.get('Content-Type', ''):
                print(f"   ✅ PDF FOUND! Size: {len(resp.content):,} bytes")
                print(f"   Content-Type: {resp.headers.get('Content-Type')}")
                break
            else:
                print(f"   ✗ {resp.status_code} - {resp.headers.get('Content-Type', 'unknown')}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
