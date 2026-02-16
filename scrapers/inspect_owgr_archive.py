"""Fetch OWGR archive page and examine download links."""

import requests
from bs4 import BeautifulSoup
from shared_utils import get_random_headers, polite_get

print("Fetching OWGR archive page...\n")

# Fetch the archive page
url = "https://www.owgr.com/archive/2025"
resp = polite_get(url, use_cache=True, delay_range=(0.5, 1.0))

if resp.status_code == 200:
    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Look for all links
    links = soup.find_all("a", href=True)
    
    print(f"Found {len(links)} total links\n")
    
    # Filter for useful ones
    download_links = []
    for link in links:
        href = link.get("href", "")
        text = link.get_text(strip=True)
        
        # Look for PDF, CSV, Excel, or download-related links
        if any(ext in href.lower() for ext in [".pdf", ".csv", ".xlsx", ".xls", "download"]):
            download_links.append((text, href))
        elif "ranking" in href.lower() or "week" in text.lower() or "download" in text.lower():
            download_links.append((text, href))
    
    if download_links:
        print(f"✓ Found {len(download_links)} potential download links:\n")
        for text, href in download_links[:20]:  # Show first 20
            print(f"  {text[:50]:50s} -> {href}")
    else:
        print("⚠️  No obvious download links found")
        print("\nAll links preview:")
        for link in links[:15]:
            print(f"  {link.get_text(strip=True)[:40]:40s} -> {link['href'][:60]}")
    
    # Check for specific patterns in the page
    print("\n" + "="*60)
    print("Searching for key patterns in HTML...")
    print("="*60)
    
    # Look for year/week selectors
    selects = soup.find_all("select")
    for select in selects:
        select_id = select.get("id", "")
        select_name = select.get("name", "")
        options = select.find_all("option")
        if options:
            print(f"\nDropdown: {select_id or select_name}")
            print(f"  Options: {len(options)} items")
            print(f"  Sample: {[opt.get_text(strip=True) for opt in options[:5]]}")
    
    # Look for button/download elements
    buttons = soup.find_all(["button", "input"], type=["button", "submit"])
    for btn in buttons:
        btn_text = btn.get_text(strip=True) or btn.get("value", "")
        if "download" in btn_text.lower() or "export" in btn_text.lower():
            print(f"\nDownload button: {btn_text}")
    
    # Save the page for manual inspection
    from pathlib import Path
    debug_file = Path("data_files/owgr_archive_2025.html")
    debug_file.write_text(resp.text, encoding="utf-8")
    print(f"\n💾 Saved page to: {debug_file}")
    
else:
    print(f"❌ Failed to fetch archive page: {resp.status_code}")

# Now try following those 302 redirects
print("\n" + "="*60)
print("Following PDF redirects...")
print("="*60)

pdf_url = "https://www.owgr.com/ranking/2026/week06.pdf"
print(f"\nTrying: {pdf_url}")

try:
    resp = requests.get(pdf_url, headers=get_random_headers(), timeout=10, allow_redirects=True)
    print(f"Final URL: {resp.url}")
    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('Content-Type')}")
    
    if resp.status_code == 200 and "pdf" in resp.headers.get('Content-Type', ''):
        print(f"✓ PDF found! Size: {len(resp.content)} bytes")
        
        # Save it
        pdf_path = Path("data_files/owgr_week06_2026.pdf")
        pdf_path.write_bytes(resp.content)
        print(f"💾 Saved to: {pdf_path}")
except Exception as e:
    print(f"Error: {e}")
