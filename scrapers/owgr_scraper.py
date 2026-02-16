"""
OWGR (Official World Golf Ranking) Scraper

Scrapes world golf rankings from owgr.com.
Coverage: 1986–present (weekly rankings).

Usage:
    # Fetch current rankings using Playwright (renders JavaScript)
    python scrapers/owgr_scraper.py --fetch --playwright --limit 200
    
    # Fetch and save to parquet
    python scrapers/owgr_scraper.py --fetch --playwright --save
    
    # Download historical archive PDFs for a year
    python scrapers/owgr_scraper.py --archive 2025 --download-pdfs data_files/owgr_pdfs
    
    # Analyze cached data
    python scrapers/owgr_scraper.py --analyze
    
    # View browser while scraping (non-headless mode)
    python scrapers/owgr_scraper.py --fetch --playwright --show-browser

Note: Playwright is required for OWGR scraping. Install with:
    pip install playwright
    playwright install

OWGR Archive PDFs:
    The OWGR site provides PDF downloads of historical rankings going back to 1986.
    Use --archive YEAR to fetch PDF links for a specific year.
"""

import argparse
import json
import re
import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup

try:
    from shared_utils import polite_get, DATA_DIR
except ImportError:
    from scrapers.shared_utils import polite_get, DATA_DIR

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright not installed. Install with: pip install playwright && playwright install")


def fetch_owgr_archive_pdfs(year: int = 2025, headless: bool = True, download_dir: Path = None) -> list:
    """
    Fetch PDF download links from OWGR archive page for a specific year.
    
    Args:
        year: Year to fetch archive for (1986-present)
        headless: Run browser in headless mode
        download_dir: Directory to save PDFs (optional)
    
    Returns:
        List of PDF URLs found
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright is required for archive downloads")
        print("   Install with: pip install playwright && playwright install")
        return []
    
    url = f"https://www.owgr.com/archive/{year}"
    
    print(f"Fetching OWGR archive PDFs for {year}...")
    print(f"URL: {url}")
    
    pdf_links = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            print("🌐 Opening archive page...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for archive content to load
            time.sleep(3)
            
            # Close cookie banner/modal if it exists
            try:
                cookie_buttons = page.query_selector_all("button:has-text('Accept'), button:has-text('OK'), button:has-text('Close'), [aria-label='Close']")
                for btn in cookie_buttons[:3]:  # Try first 3 found
                    try:
                        print("🍪 Closing banner/modal...")
                        btn.click(timeout=2000)
                        time.sleep(1)
                        break
                    except:
                        pass
            except Exception as e:
                print(f"   No banner to close")
            
            # Select the correct year from the dropdown
            print(f"📅 Selecting year {year} from dropdown...")
            try:
                # Click the year dropdown to open it
                dropdown = page.query_selector("div.custom__select__control")
                if dropdown:
                    dropdown.click()
                    time.sleep(1)
                    
                    # Find and click the option for the target year
                    # Options appear in a menu with specific class
                    year_option = page.query_selector(f"div[id*='react-select'][id*='option']:has-text('{year}')")
                    if year_option:
                        print(f"   ✓ Found year {year} option, click ing...")
                        year_option.click()
                        time.sleep(3)  # Wait for React to reload content
                    else:
                        print(f"   ⚠️  Year {year} option not found in dropdown")
                else:
                    print(f"   ⚠️  Dropdown not found")
            except Exception as e:
                print(f"   ⚠️  Error selecting year: {e}")
            
            # Look for archive items (they might have a specific class or data attribute)
            print("⏳ Waiting for archive items to load...")
            try:
                # Wait for archive content elements to appear
                page.wait_for_selector("a[href*='.pdf'], a[href*='owgr'], div.archivePageComponent", timeout=10000)
                time.sleep(2)
            except Exception as e:
                print(f"⚠️  Timeout waiting for archive elements: {e}")
            
            # Look for "See more" button and click it multiple times
            max_clicks = 5
            for i in range(max_clicks):
                try:
                    see_more_button = page.query_selector("div.archivePageComponent_more__oJ_QL, button:has-text('See more'), a:has-text('See more')")
                    if see_more_button and see_more_button.is_visible():
                        print(f"👆 Clicking 'See more' button ({i+1}/{max_clicks})...")
                        see_more_button.click()
                        time.sleep(2)  # Wait for content to load
                    else:
                        break
                except Exception as e:
                    print(f"⚠️  No more 'See more' button found")
                    break
            
# Save HTML for debugging
            from pathlib import Path
            html = page.content()
            debug_path = Path(f"data_files/owgr_archive_{year}_rendered.html")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(html, encoding="utf-8")
            print(f"💾 Saved rendered HTML to: {debug_path}")
            
            # Set up network request interception to capture PDF URLs
            pdf_urls = []
            def capture_pdf_request(route, request):
                url = request.url
                if '.pdf' in url or 'download' in url.lower():
                    print(f"   📄 Captured PDF request: {url}")
                    pdf_urls.append(url)
                route.continue_()
            
            # Intercept requests for PDFs
            #page.route("**/*", capture_pdf_request)
            
            # Alternative: listen for downloads
            download_promises = []
            
            print("🔎 Finding download buttons...")
            # Find all "Download rankings" buttons
            download_buttons = page.query_selector_all("div.archiveItemComponent_download__ju9MG")
            print(f"   Found {len(download_buttons)} download buttons")
            
            if not download_buttons:
                # Try alternative selector
                download_buttons = page.query_selector_all("div:has-text('Download rankings')")
                print(f"   Found {len(download_buttons)} buttons (alternative selector)")
            
            # Parse HTML to get week numbers
            soup = BeautifulSoup(html, "html.parser")
            week_elements = soup.find_all("div", class_=lambda c: c and "archiveItemComponent_title__second" in c)
            week_numbers = []
            for elem in week_elements:
                week_text = elem.get_text(strip=True)
                if "Week" in week_text:
                    try:
                        week_num = int(week_text.replace("Week", "").strip())
                        week_numbers.append((week_num, week_text))
                    except:
                        pass
            
            print(f"   Found {len(week_numbers)} weeks: {[w[0] for w in week_numbers]}")
            
            # Each week has 2 download buttons: rankings + federation rankings
            # Group buttons by pairs (assuming they're sequential)
            buttons_per_week = 2
            
            for week_idx, (week_num, week_text) in enumerate(week_numbers):
                # Get both buttons for this week
                button_start = week_idx * buttons_per_week
                button_end = button_start + buttons_per_week
                week_buttons = download_buttons[button_start:button_end]
                
                if not week_buttons:
                    continue
                
                print(f"   📅 Week {week_num:02d} ({len(week_buttons)} buttons)")
                
                for btn_idx, button in enumerate(week_buttons):
                    try:
                        # Set up response listener before clicking
                        captured_url = []
                        def handle_response(response):
                            url = response.url
                            if '.pdf' in url.lower() or response.headers.get('content-type', '').startswith('application/pdf'):
                                print(f"      📥 PDF response: {url}")
                                captured_url.append(url)
                        
                        page.on("response", handle_response)
                        
                        # Scroll button into view to ensure it's clickable
                        button.scroll_into_view_if_needed()
                        time.sleep(0.3)
                        
                        button_type = "rankings" if btn_idx == 0 else "federation"
                        print(f"      🖱️  Clicking {button_type} button...")
                        
                        # Force click (bypass visibility checks)
                        button.click(force=True, timeout=5000)
                        time.sleep(1)  # Wait for response
                        
                        page.remove_listener("response", handle_response)
                        
                        if captured_url:
                            url = captured_url[0]
                            # Extract actual filename from URL
                            import urllib.parse
                            parsed_url = urllib.parse.urlparse(url)
                            filename = parsed_url.path.split('/')[-1]
                            filename = urllib.parse.unquote(filename)  # Decode %20 etc.
                            
                            pdf_links.append({
                                "url": url,
                                "text": f"{week_text} - {button_type.title()}",
                                "filename": filename,
                                "year": year,
                                "week": week_num,
                                "type": button_type
                            })
                            print(f"         ✓ Captured: {filename}")
                        
                    except Exception as e:
                        print(f"         ⚠️  Failed: {str(e)[:50]}")
            
            browser.close()
        
        print(f"✓ Found {len(pdf_links)} PDF links")
        
        # Optionally download PDFs
        if download_dir and pdf_links:
            download_dir = Path(download_dir)
            download_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📥 Downloading PDFs to {download_dir}...")
            for i, pdf_info in enumerate(pdf_links, 1):
                try:
                    import requests
                    resp = requests.get(pdf_info["url"], timeout=30)
                    if resp.status_code == 200:
                        # Use the filename we already extracted and decoded
                        filename = pdf_info.get("filename", f"owgr_{year}_week_{pdf_info['week']:02d}_{pdf_info['type']}.pdf")
                        
                        # Sanitize filename for Windows/Unix compatibility
                        filename = filename.replace("/", "_").replace("\\", "_").replace(":", "-")
                        
                        # Prefix with year to avoid overwrites when downloading multiple years
                        filename = f"{year}_{filename}"
                        
                        filepath = download_dir / filename
                        filepath.write_bytes(resp.content)
                        file_size = len(resp.content) / 1024  # KB
                        print(f"  {i}/{len(pdf_links)}: ✓ {filename} ({file_size:.1f} KB)")
                    else:
                        print(f"  {i}/{len(pdf_links)}: ✗ Failed ({resp.status_code}) - {pdf_info.get('filename', 'unknown')}")
                except Exception as e:
                    print(f"  {i}/{len(pdf_links)}: ✗ Error: {pdf_info.get('filename', 'unknown')} - {str(e)[:50]}")
            
            print(f"\n💾 Downloaded {len(pdf_links)} PDFs")
        
        return pdf_links
    
    except Exception as e:
        print(f"❌ Error fetching archive PDFs: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_owgr_rankings_playwright(page_size: int = 100, max_players: int = 200, headless: bool = True) -> pd.DataFrame:
    """
    Fetch OWGR rankings using Playwright to render JavaScript.
    
    Args:
        page_size: Number of players per page (via URL param)
        max_players: Maximum number of players to fetch
        headless: Run browser in headless mode
    
    Returns:
        DataFrame with ranking data
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("❌ Playwright is not installed. Install with:")
        print("   pip install playwright")
        print("   playwright install")
        return pd.DataFrame()
    
    base_url = "https://www.owgr.com/ranking"
    params_str = f"?pageNo=1&pageSize={page_size}&country=All"
    url = base_url + params_str
    
    print(f"Fetching OWGR rankings with Playwright (limit: {max_players})...")
    print(f"URL: {url}")
    
    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            print("🌐 Opening page...")
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for table to load
            print("⏳ Waiting for table to load...")
            try:
                page.wait_for_selector("table", timeout=15000)
                print("✓ Table loaded")
            except PlaywrightTimeout:
                print("⚠️  Table did not load in time")
            
            # Additional wait for JavaScript to populate data
            time.sleep(2)
            
            # Get the rendered HTML
            html = page.content()
            
            browser.close()
            
            # Parse the rendered HTML
            print("📊 Parsing table data...")
            df = parse_owgr_table_html(html)
            
            # Limit to max_players
            if not df.empty and len(df) > max_players:
                df = df.head(max_players)
            
            return df
    
    except Exception as e:
        print(f"❌ Error fetching OWGR with Playwright: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_owgr_rankings(page_size: int = 100, max_players: int = 200, force_refresh: bool = False, save_debug: bool = False, use_playwright: bool = True) -> pd.DataFrame:
    """
    Fetch current OWGR rankings.
    
    Args:
        page_size: Number of players per page
        max_players: Maximum number of players to fetch
        force_refresh: If True, bypass cache and re-download
        save_debug: If True, save JSON structure for debugging
        use_playwright: If True, use Playwright to render JavaScript (recommended)
    
    Returns:
        DataFrame with ranking data
    """
    # Use Playwright by default if available
    if use_playwright and PLAYWRIGHT_AVAILABLE:
        return fetch_owgr_rankings_playwright(page_size, max_players)
    
    # Fallback to simple HTTP (won't work for OWGR but kept for compatibility)
    base_url = "https://www.owgr.com/ranking"
    params_str = f"?pageNo=1&pageSize={page_size}&country=All"
    url = base_url + params_str
    
    print(f"Fetching OWGR rankings (limit: {max_players})...")
    print("⚠️  Note: OWGR requires JavaScript. Use --playwright flag for full data.")
    
    try:
        # Fetch with caching
        resp = polite_get(url, use_cache=True, force_refresh=force_refresh)
        
        # Try parsing as JSON first (in case it's an API endpoint)
        try:
            data = resp.json()
            print("✓ Received JSON response")
            return parse_owgr_json(data)
        except json.JSONDecodeError:
            # It's HTML, parse it
            print("✓ Received HTML response, parsing...")
            return parse_owgr_html(resp.text, save_debug=save_debug)
    
    except Exception as e:
        print(f"❌ Error fetching OWGR: {e}")
        return pd.DataFrame()


def parse_owgr_json(data: dict) -> pd.DataFrame:
    """Parse OWGR data from JSON response."""
    rows = []
    
    # The structure depends on the actual API - this is a best guess
    # We'll need to inspect the actual response to refine this
    
    if isinstance(data, list):
        # Array of players
        for player in data:
            rows.append({
                "rank": player.get("rank"),
                "player_name": player.get("name") or player.get("playerName"),
                "country": player.get("country"),
                "avg_points": player.get("avgPoints") or player.get("average"),
                "total_points": player.get("totalPoints") or player.get("total"),
                "events_played": player.get("eventsPlayed") or player.get("events"),
            })
    elif isinstance(data, dict):
        # Check for common JSON structures
        if "rankings" in data:
            for player in data["rankings"]:
                rows.append({
                    "rank": player.get("rank"),
                    "player_name": player.get("name") or player.get("playerName"),
                    "country": player.get("country"),
                    "avg_points": player.get("avgPoints") or player.get("average"),
                    "total_points": player.get("totalPoints") or player.get("total"),
                    "events_played": player.get("eventsPlayed") or player.get("events"),
                })
        elif "data" in data:
            for player in data["data"]:
                rows.append({
                    "rank": player.get("rank"),
                    "player_name": player.get("name") or player.get("playerName"),
                    "country": player.get("country"),
                    "avg_points": player.get("avgPoints") or player.get("average"),
                    "total_points": player.get("totalPoints") or player.get("total"),
                    "events_played": player.get("eventsPlayed") or player.get("events"),
                })
    
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} players from JSON")
    return df


def parse_owgr_table_html(html: str) -> pd.DataFrame:
    """
    Parse OWGR ranking table from fully-rendered HTML.
    This function is designed for HTML after JavaScript execution.
    
    Args:
        html: Fully rendered HTML content
    
    Returns:
        DataFrame with ranking data
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Find the main ranking table
    table = soup.find("table")
    
    if not table:
        print("⚠️  No table found in rendered HTML")
        return pd.DataFrame()
    
    rows = []
    
    # Extract table headers
    headers = []
    thead = table.find("thead")
    if thead:
        header_row = thead.find("tr")
        if header_row:
            for th in header_row.find_all(["th", "td"]):
                header_text = th.get_text(strip=True)
                # Clean up header text
                header_text = header_text.lower().replace("\\n", " ").strip()
                headers.append(header_text)
    
    print(f"Table headers: {headers}")
    
    # Extract data rows
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr", recursive=False):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cell_text = td.get_text(strip=True)
                # Clean up cell text (remove extra whitespace, newlines)
                cell_text = " ".join(cell_text.split())
                cells.append(cell_text)
            
            if cells:  # Only add non-empty rows
                rows.append(cells)
    
    if not rows:
        print("⚠️  No data rows found in table")
        return pd.DataFrame()
    
    # Create DataFrame
    if headers and len(headers) > 0:
        # Handle case where rows might have different lengths
        max_cols = max(len(row) for row in rows)
        
        # Pad headers if needed
        while len(headers) < max_cols:
            headers.append(f"col_{len(headers)}")
        
        # Pad rows if needed
        for i, row in enumerate(rows):
            while len(row) < len(headers):
                row.append("")
        
        df = pd.DataFrame(rows, columns=headers[:len(rows[0]) if rows else 0])
    else:
        # Use generic column names
        df = pd.DataFrame(rows)
        if not df.empty:
            df.columns = [f"col_{i}" for i in range(len(df.columns))]
    
    # Clean up DataFrame
    # Remove rows that are completely empty or just dashes
    df = df.replace("-", pd.NA)
    df = df.dropna(how="all")
    
    # Try to identify and rename common columns
    column_mapping = {
        "ranking": "rank",
        "this week": "rank",
        "ctry": "country",
        "name": "player_name",
        "average points": "avg_points",
        "total points": "total_points",
        "events played (divisor)": "events_divisor",
        "events played (actual)": "events_actual",
        "points lost (2026)": "points_lost",
        "points won (2026)": "points_won",
        "last week": "rank_last_week",
        "end 2025": "rank_end_2025",
    }
    
    df.columns = [column_mapping.get(col.lower(), col) for col in df.columns]
    
    print(f"✓ Parsed {len(df)} players from table")
    print(f"Columns: {list(df.columns)}")
    
    return df


def extract_json_from_script(html: str, save_debug: bool = False) -> dict:
    """Extract JSON data from Next.js script tags."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Look for script tags with id="__NEXT_DATA__" (Next.js pattern)
    next_data = soup.find("script", {"id": "__NEXT_DATA__"})
    if next_data:
        try:
            data = json.loads(next_data.string)
            print("✓ Found __NEXT_DATA__ script")
            
            if save_debug:
                debug_file = DATA_DIR / "owgr_next_data_debug.json"
                debug_file.write_text(json.dumps(data, indent=2))
                print(f"💾 Saved JSON structure to: {debug_file}")
            
            return data
        except json.JSONDecodeError:
            pass
    
    # Look for other script tags with JSON
    scripts = soup.find_all("script")
    for script in scripts:
        if script.string and "{" in script.string:
            # Try to find JSON objects
            text = script.string.strip()
            if text.startswith("{") and "ranking" in text.lower():
                try:
                    data = json.loads(text)
                    print("✓ Found JSON in script tag")
                    return data
                except json.JSONDecodeError:
                    # Might be embedded in JavaScript, try to extract
                    import re
                    # Look for patterns like: var data = {...}
                    match = re.search(r'(?:var|const|let)\s+\w+\s*=\s*(\{.+\})', text, re.DOTALL)
                    if match:
                        try:
                            data = json.loads(match.group(1))
                            print("✓ Extracted JSON from JavaScript variable")
                            return data
                        except json.JSONDecodeError:
                            pass
    
    return {}


def parse_owgr_html(html: str, save_debug: bool = False) -> pd.DataFrame:
    """Parse OWGR data from HTML response."""
    # First, try to extract JSON from script tags (Next.js sites)
    json_data = extract_json_from_script(html, save_debug=save_debug)
    
    if json_data:
        print("Attempting to parse from embedded JSON...")
        df = parse_owgr_json_nextjs(json_data)
        if not df.empty:
            return df
    
    # Fallback to HTML table parsing
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    
    # Look for table with class containing 'ranking' or 'table'
    tables = soup.find_all("table")
    
    if not tables:
        print("⚠️  No tables found in HTML")
        print("HTML preview (first 500 chars):")
        print(html[:500])
        return pd.DataFrame()
    
    print(f"Found {len(tables)} table(s)")
    
    # Try the first table
    table = tables[0]
    
    # Get headers
    headers = []
    header_row = table.find("thead")
    if header_row:
        for th in header_row.find_all("th"):
            headers.append(th.get_text(strip=True).lower())
    
    if not headers:
        # Try first row as header
        first_row = table.find("tr")
        if first_row:
            for th in first_row.find_all(["th", "td"]):
                headers.append(th.get_text(strip=True).lower())
    
    print(f"Headers: {headers}")
    
    # Get data rows
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells and len(cells) >= 2:  # At least rank and name
            rows.append(cells)
    
    if not rows:
        print("⚠️  No data rows found")
        return pd.DataFrame()
    
    # Create DataFrame
    if headers and len(headers) == len(rows[0]):
        df = pd.DataFrame(rows, columns=headers)
    else:
        # Generic column names
        df = pd.DataFrame(rows)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
    
    print(f"Parsed {len(df)} players from HTML")
    return df


def parse_owgr_json_nextjs(data: dict) -> pd.DataFrame:
    """Parse OWGR data from Next.js embedded JSON."""
    rows = []
    
    # Navigate through Next.js data structure
    # Common paths: props.pageProps.data, props.pageProps.rankings, etc.
    
    def find_rankings(obj, path=""):
        """Recursively search for ranking data."""
        if isinstance(obj, dict):
            # Look for keys that might contain rankings
            for key in ["rankings", "data", "players", "list", "items"]:
                if key in obj and isinstance(obj[key], list):
                    print(f"Found rankings at: {path}.{key}")
                    return obj[key]
            # Recurse into nested dicts
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    result = find_rankings(value, f"{path}.{key}")
                    if result:
                        return result
        elif isinstance(obj, list) and obj:
            # Check if this looks like ranking data
            first_item = obj[0]
            if isinstance(first_item, dict):
                # Check for ranking-like keys
                keys = set(first_item.keys())
                ranking_keys = {"rank", "ranking", "position", "name", "player", "points"}
                if keys & ranking_keys:
                    print(f"Found rankings at: {path} (list)")
                    return obj
        return None
    
    rankings_list = find_rankings(data)
    
    if not rankings_list:
        print("⚠️  Could not find rankings in JSON structure")
        # Print structure for debugging
        print("JSON keys:", list(data.keys()) if isinstance(data, dict) else "not a dict")
        return pd.DataFrame()
    
    # Parse the rankings list
    for item in rankings_list:
        if isinstance(item, dict):
            # Extract common fields with various possible key names
            row = {
                "rank": (item.get("rank") or item.get("ranking") or 
                        item.get("position") or item.get("thisWeek")),
                "player_name": (item.get("name") or item.get("playerName") or 
                               item.get("player") or item.get("fullName")),
                "country": (item.get("country") or item.get("nationality") or
                           item.get("countryCode")),
                "avg_points": (item.get("avgPoints") or item.get("average") or
                              item.get("averagePoints")),
                "total_points": (item.get("totalPoints") or item.get("total") or
                                item.get("points")),
                "events_played": (item.get("eventsPlayed") or item.get("events") or
                                 item.get("divisor")),
            }
            
            # Add any other fields that exist
            for key, value in item.items():
                if key not in row:
                    row[key] = value
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} players from Next.js JSON")
    return df


def analyze_cached_owgr() -> dict:
    """
    Analyze cached OWGR data to understand structure.
    
    Returns:
        Dict with analysis results
    """
    from shared_utils import CACHE_DIR
    import hashlib
    
    # Find OWGR cached files
    owgr_url_base = "https://www.owgr.com/ranking"
    cache_hash = hashlib.md5(owgr_url_base.encode()).hexdigest()
    
    # Look for any cached files that might be OWGR
    owgr_cache_files = []
    for cache_file in CACHE_DIR.glob("*.html"):
        content = cache_file.read_text(encoding="utf-8", errors="ignore")
        if "owgr" in content.lower() or "world golf ranking" in content.lower():
            owgr_cache_files.append(cache_file)
    
    print(f"\nFound {len(owgr_cache_files)} cached OWGR file(s)")
    
    analysis = {
        "cached_files": len(owgr_cache_files),
        "samples": []
    }
    
    for cache_file in owgr_cache_files[:3]:  # Analyze first 3
        print(f"\n📄 Analyzing: {cache_file.name}")
        content = cache_file.read_text(encoding="utf-8", errors="ignore")
        
        # Try JSON
        try:
            data = json.loads(content)
            analysis["samples"].append({
                "file": cache_file.name,
                "type": "json",
                "keys": list(data.keys()) if isinstance(data, dict) else "list",
                "sample": str(data)[:500]
            })
            print(f"  Type: JSON")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
            print(f"  Preview: {str(data)[:200]}...")
        except json.JSONDecodeError:
            # Parse as HTML
            soup = BeautifulSoup(content, "html.parser")
            tables = soup.find_all("table")
            scripts = soup.find_all("script")
            
            analysis["samples"].append({
                "file": cache_file.name,
                "type": "html",
                "tables": len(tables),
                "scripts": len(scripts),
                "sample": content[:500]
            })
            
            print(f"  Type: HTML")
            print(f"  Tables found: {len(tables)}")
            print(f"  Script tags: {len(scripts)}")
            
            # Look for JSON in script tags
            for script in scripts:
                script_text = script.get_text()
                if "ranking" in script_text.lower() and "{" in script_text:
                    print(f"  ⚠️  Found JSON data in <script> tag")
                    print(f"  Preview: {script_text[:200]}...")
                    break
    
    return analysis


def save_owgr_data(df: pd.DataFrame, filename: str = "owgr_rankings.parquet"):
    """Save OWGR data to parquet file."""
    output_path = DATA_DIR / filename
    df.to_parquet(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Rows: {len(df)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="OWGR Scraper")
    parser.add_argument("--fetch", action="store_true", help="Fetch current rankings")
    parser.add_argument("--archive", type=int, metavar="YEAR", help="Fetch archive PDFs for specific year (1986-2026)")
    parser.add_argument("--analyze", action="store_true", help="Analyze cached data")
    parser.add_argument("--limit", type=int, default=200, help="Max players to fetch (default: 200)")
    parser.add_argument("--save", action="store_true", help="Save to parquet file")
    parser.add_argument("--debug", action="store_true", help="Save JSON structure for debugging")
    parser.add_argument("--force", action="store_true", help="Force refresh (bypass cache)")
    parser.add_argument("--playwright", action="store_true", default=True, help="Use Playwright to render JavaScript (default)")
    parser.add_argument("--no-playwright", action="store_true", help="Don't use Playwright (fallback mode)")
    parser.add_argument("--show-browser", action="store_true", help="Show browser window (non-headless mode)")
    parser.add_argument("--download-pdfs", type=str, metavar="DIR", help="Download archive PDFs to directory")
    
    args = parser.parse_args()
    
    # Handle archive mode
    if args.archive:
        download_dir = Path(args.download_pdfs) if args.download_pdfs else None
        headless = not args.show_browser if hasattr(args, 'show_browser') else True
        
        pdf_links = fetch_owgr_archive_pdfs(
            year=args.archive,
            headless=headless,
            download_dir=download_dir
        )
        
        if pdf_links:
            print("\n" + "="*60)
            print(f"ARCHIVE PDF LINKS ({args.archive})")
            print("="*60)
            for i, pdf in enumerate(pdf_links[:10], 1):
                print(f"{i:3d}. {pdf['text'][:50]:50s} -> {pdf['url']}")
            if len(pdf_links) > 10:
                print(f"     ... and {len(pdf_links) - 10} more")
        
        return
    
    if args.analyze:
        analysis = analyze_cached_owgr()
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(json.dumps(analysis, indent=2))
    
    if args.fetch:
        use_playwright = args.playwright and not args.no_playwright
        
        # Override headless mode if show-browser is set
        if use_playwright and hasattr(args, 'show_browser') and args.show_browser:
            if PLAYWRIGHT_AVAILABLE:
                df = fetch_owgr_rankings_playwright(
                    page_size=args.limit if args.limit <= 200 else 200,
                    max_players=args.limit,
                    headless=False
                )
            else:
                print("⚠️  Playwright not available, falling back to HTTP mode")
                df = fetch_owgr_rankings(
                    max_players=args.limit,
                    force_refresh=args.force if hasattr(args, 'force') else False,
                    save_debug=args.debug if hasattr(args, 'debug') else False,
                    use_playwright=False
                )
        else:
            df = fetch_owgr_rankings(
                max_players=args.limit,
                force_refresh=args.force if hasattr(args, 'force') else False,
                save_debug=args.debug if hasattr(args, 'debug') else False,
                use_playwright=use_playwright
            )
        
        if not df.empty:
            print("\n" + "="*60)
            print("FETCHED DATA PREVIEW")
            print("="*60)
            print(df.head(10))
            print(f"\nShape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            if args.save:
                save_owgr_data(df)
        else:
            print("\n⚠️  No data fetched. Try --analyze to inspect cached files.")
    
    if not args.fetch and not args.analyze and not args.archive:
        print("Usage: python scrapers/owgr_scraper.py [OPTIONS]")
        print("\nOptions:")
        print("  --fetch            Fetch current OWGR rankings")
        print("  --archive YEAR     Fetch archive PDFs for specific year (1986-2026)")
        print("  --download-pdfs DIR  Download PDFs to directory (use with --archive)")
        print("  --playwright       Use Playwright to render JavaScript (default)")
        print("  --no-playwright    Disable Playwright (basic HTTP only)")
        print("  --show-browser     Show browser window while scraping")
        print("  --analyze          Analyze cached OWGR data")
        print("  --limit N          Limit to N players (default: 200)")
        print("  --save             Save to parquet file")
        print("  --force            Force refresh (bypass cache)")
        print("  --debug            Save JSON structure for debugging")
        print("\nExamples:")
        print("  # Fetch current rankings")
        print("  python scrapers/owgr_scraper.py --fetch --playwright --save")
        print("\n  # Download 2025 archive PDFs")
        print("  python scrapers/owgr_scraper.py --archive 2025 --download-pdfs data_files/owgr_pdfs")
        print("\nPlaywright setup:")
        print("  pip install playwright")
        print("  playwright install")


if __name__ == "__main__":
    main()
