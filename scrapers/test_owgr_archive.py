"""Test OWGR archive download links."""

import requests
from shared_utils import get_random_headers
from datetime import datetime, timedelta

print("Testing OWGR archive download patterns...\n")

# Common patterns for golf ranking archives
base_patterns = [
    "https://www.owgr.com/archive/{year}",
    "https://www.owgr.com/downloads/archive",
    "https://www.owgr.com/downloads/{year}",
    "https://www.owgr.com/ranking/archive",
    "https://www.owgr.com/api/archive",
]

# Test various endpoints
for pattern in base_patterns:
    if "{year}" in pattern:
        url = pattern.format(year=2024)
    else:
        url = pattern
    
    print(f"Trying: {url}")
    try:
        resp = requests.get(url, headers=get_random_headers(), timeout=10, allow_redirects=True)
        print(f"  Status: {resp.status_code}")
        
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            print(f"  Content-Type: {content_type}")
            
            # Check if it's a PDF
            if "pdf" in content_type:
                print(f"  ✓ PDF found!")
                print(f"  Size: {len(resp.content)} bytes")
            
            # Check if it's HTML with download links
            elif "html" in content_type:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for PDF/CSV download links
                links = soup.find_all("a", href=True)
                pdf_links = [a for a in links if ".pdf" in a["href"].lower()]
                csv_links = [a for a in links if ".csv" in a["href"].lower()]
                xlsx_links = [a for a in links if ".xlsx" in a["href"].lower() or ".xls" in a["href"].lower()]
                
                if pdf_links:
                    print(f"  ✓ Found {len(pdf_links)} PDF links:")
                    for link in pdf_links[:5]:
                        print(f"    - {link.get('href')}")
                
                if csv_links:
                    print(f"  ✓ Found {len(csv_links)} CSV links:")
                    for link in csv_links[:5]:
                        print(f"    - {link.get('href')}")
                
                if xlsx_links:
                    print(f"  ✓ Found {len(xlsx_links)} Excel links:")
                    for link in xlsx_links[:5]:
                        print(f"    - {link.get('href')}")
        
        elif resp.status_code in [301, 302, 307, 308]:
            print(f"  Redirect to: {resp.headers.get('Location')}")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    print()

# Try direct PDF download pattern (common for weekly rankings)
print("\nTrying weekly ranking PDF pattern...")
# OWGR releases rankings on Mondays
recent_date = datetime.now() - timedelta(days=7)
year = recent_date.year
week = recent_date.isocalendar()[1]

pdf_patterns = [
    f"https://www.owgr.com/ranking/{year}/week{week:02d}.pdf",
    f"https://www.owgr.com/downloads/ranking_{year}_{week:02d}.pdf",
    f"https://www.owgr.com/downloads/ranking_{year}.pdf",
]

for url in pdf_patterns:
    print(f"Trying: {url}")
    try:
        resp = requests.head(url, headers=get_random_headers(), timeout=5)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"  ✓ Found! Content-Type: {resp.headers.get('Content-Type')}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
