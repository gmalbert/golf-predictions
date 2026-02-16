# OWGR Scraping Analysis Report

**Date:** February 14, 2026  
**Site:** https://www.owgr.com

## Summary

OWGR (Official World Golf Ranking) uses a modern Next.js/React application that loads ranking data **dynamically via JavaScript**. Simple HTTP requests do not return the full ranking data.

**IMPORTANT FINDING:** OWGR provides **PDF downloads** of historical rankings dating back to 1986 at `https://www.owgr.com/archive/{year}`. This is the **easiest way** to get historical data rather than scraping live rankings.

---

## What Data is Available

Based on the HTML structure observed, OWGR provides comprehensive ranking data including:

### Fields Available:
1. **Ranking** - Current world ranking position
2. **Last Week** - Previous week's ranking
3. **End 2025** - Ranking at end of 2025
4. **Country** (ctry) - Player's country code
5. **Name** - Player's full name
6. **Average Points** - Average ranking points
7. **Total Points** - Total ranking points accumulated
8. **Events Played (Divisor)** - Number of events used for averaging
9. **Points Lost (2026)** - Points that will drop off in 2026
10. **Points Won (2026)** - Points gained in 2026
11. **Events Played (Actual)** - Total events actually played

### Coverage:
- **Historical:** 1986–present (weekly rankings)
- **Update Frequency:** Weekly (typically Monday)

---

## Technical Challenges Found

### 1. Dynamic JavaScript Loading ❌
- The main HTML page contains only the page structure
- Ranking data is loaded client-side after page render
- Standard HTTP requests (requests/BeautifulSoup) cannot capture the data

### 2. No Public API ❌
Tested multiple endpoint patterns, all returned 404:
- `https://www.owgr.com/api/v1/ranking`
- `https://www.owgr.com/api/rankings`
- `https://www.owgr.com/_next/data/{buildId}/ranking.json`
- `https://api.owgr.com/v1/ranking`

### 3. Next.js Data Structure
- Site uses Next.js with server-side rendering
- `__NEXT_DATA__` script tag contains page metadata only
- Actual ranking data not embedded in initial page load

---

## Recommended Approaches

### Option 1: Download Historical PDFs (EASIEST) ✅✅ **WORKING**

**DISCOVERY:** OWGR provides PDF downloads of Top 300 rankings going back to 1986!

**URL:** `https://www.owgr.com/archive/{year}`  
**Coverage:** 1986–present  
**Format:** Weekly PDF files (both regular rankings and federation rankings)

**Implementation Status:** ✅ **FULLY WORKING** (as of Feb 2026)

**How it works:**
1. Playwright navigates to archive page
2. Closes cookie banners automatically
3. Extracts week numbers from rendered HTML
4. Clicks download buttons while intercepting network requests
5. Captures actual PDF URLs from CDN (assets-us-01.kc-usercontent.com)
6. Downloads all PDFs with original filenames

**Pros:**
- Official data source
- Complete historical archive (1986-2026)
- Reliable and maintained by OWGR
- Both regular and federation rankings available
- **Successfully tested:** Downloads 11 PDFs from 2026 archive in ~30 seconds

**Cons:**
- Need to parse PDF files (next step)
- Requires Playwright to get download links (dynamically loaded)

**Working Implementation:**
```bash
# Get PDF links for any year (1986-2026)
python scrapers/owgr_scraper.py --archive 2024

# Download all PDFs for a specific year
python scrapers/owgr_scraper.py --archive 2024 --download-pdfs data_files/owgr_pdfs

# Show browser while scraping (for debugging)
python scrapers/owgr_scraper.py --archive 2024 --show-browser

# Batch download multiple years (PowerShell)
2020..2025 | ForEach-Object { 
    python scrapers/owgr_scraper.py --archive $_ --download-pdfs data_files/owgr_pdfs 
}
```

**Downloaded Files:**
- Main rankings: `owgr01f2026.pdf`, `owgr02f2026.pdf`, etc. (779-5500 KB)
- Federation rankings: `Federation Ranking List {date} - Week {N}.pdf` (167-232 KB)

**Next Steps (PDF Parsing):**
1. Install PDF parsing library: `pip install pdfplumber tabula-py`
2. Extract tables from PDFs → structured data
3. Convert to Parquet format for efficient storage
4. Build complete historical database (1986-present)

**Example successful run:**
```
python scrapers/owgr_scraper.py --archive 2026 --download-pdfs data_files/owgr_pdfs

✓ Found 11 PDF links
📥 Downloading PDFs to data_files\owgr_pdfs...
  1/11: ✓ owgr01f2026.pdf (779.7 KB)
  2/11: ✓ Federation Ranking List 4th January 2025 - Week 01.pdf (167.5 KB)
  ...
  11/11: ✓ owgr06f2026.pdf (797.4 KB)
💾 Downloaded 11 PDFs
```

### Option 2: Browser Automation (CURRENT RANKINGS) ✅

Use Selenium or Playwright to render the JavaScript and extract data.

**Pros:**
- Can access all data visible on the page
- Handles JavaScript rendering
- Most reliable method

**Cons:**
- Slower than API calls
- Requires browser driver
- More resource-intensive

**Implementation:**
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

driver.get('https://www.owgr.com/ranking')

# Wait for table to load
wait = WebDriverWait(driver, 10)
table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

# Extract table data
rows = table.find_elements(By.TAG_NAME, "tr")
for row in rows:
    cells = row.find_elements(By.TAG_NAME, "td")
    # Process cells...

driver.quit()
```

### Option 2: Network Traffic Analysis 🔍

Inspect the browser's network tab to find the actual API endpoint used by JavaScript.

**Steps:**
1. Open https://www.owgr.com/ranking in browser
2. Open Developer Tools → Network tab
3. Filter by XHR/Fetch requests
4. Look for JSON responses containing ranking data
5. Extract the API endpoint and parameters

**Likely patterns:**
- GraphQL endpoint
- REST API with authentication tokens
- WebSocket connection

### Option 3: Downloadable Files 📥

Check if OWGR provides CSV/Excel downloads.

**Checked:**
- `/downloads` - needs verification
- Historical data archives
- Weekly ranking exports

### Option 4: Alternative Data Sources ✅

Consider using other sources that aggregate OWGR data:

- **DataGolf** - Provides OWGR data via their API (paid)
- **ESPN API** - May include world rankings in player profiles
- **Sports Reference / Golf-Reference** - Historical rankings
- **Data.gov / Sports datasets** - Archived OWGR data

---

## Current Scraper Status

Created `scrapers/owgr_scraper.py` with:
- ✅ Caching system (uses `data_files/cache/`)
- ✅ HTML parsing framework
- ✅ Next.js JSON extraction
- ❌ Does not yet extract live data (requires browser automation)

---

## Next Steps

### Immediate (for basic functionality):
1. Add Selenium/Playwright support to `owgr_scraper.py`
2. Update `requirements.txt` with `selenium` or `playwright`
3. Implement table extraction after JavaScript render

### Short-term:
1. Inspect network traffic to find undocumented API
2. Check for downloadable CSV/Excel files
3. Add historical data fetching (weekly snapshots)

### Long-term:
1. Build a local database of OWGR history
2. Set up weekly automated scraping
3. Add player profile scraping (detailed stats per player)

---

## Usage Recommendation

**Best approach:** Download historical PDFs from the archive (1986-present), then use browser automation for current/live rankings.

**For historical data (recommended):**
```bash
# Download PDFs for years you need
python scrapers/owgr_scraper.py --archive 2024 --download-pdfs data_files/owgr_pdfs
python scrapers/owgr_scraper.py --archive 2023 --download-pdfs data_files/owgr_pdfs
```

**For current rankings:**
```python
# Fetch live rankings with Playwright
python scrapers/owgr_scraper.py --fetch --playwright --save
```

**ESPN Alternative (includes OWGR):**
```python
# ESPN includes OWGR in player profiles
url = f"https://site.api.espn.com/apis/common/v3/sports/golf/pga/athletes/{player_id}"
resp = requests.get(url)
data = resp.json()
owgr_rank = data.get("athlete", {}).get("worldRank")
```

**DataGolf (free tier):**
- View rankings at https://datagolf.com/rankings
- Limited API access for free tier

---

## Files Created

1. **`scrapers/owgr_scraper.py`** - Main scraper with Playwright support ✅ WORKING
   - PDF archive downloads via network interception
   - Live rankings scraping with browser automation
   - Automatic cookie banner handling
   - CLI with `--archive`, `--download-pdfs`, `--fetch` modes

2. `scrapers/debug_owgr_json.py` - JSON structure inspector
3. `scrapers/test_owgr_endpoints.py` - API endpoint tester
4. `scrapers/test_nextjs_data.py` - Next.js data endpoint tester

---

## Conclusion

OWGR data is **available in two forms**:

1. ✅ **Historical PDFs** (1986–present) - Successfully implemented via network interception
2. ✅ **Live rankings** - Playwright browser automation implemented

**Successful Implementation (Feb 2026):**

✅ **PDF Archive Downloads (COMPLETE)**
- Playwright navigates to `/archive/{year}` pages
- Closes cookie/consent banners automatically
- Intercepts network requests to capture CDN URLs
- Downloads both regular and federation rankings
- Tested and working for 2026 (11 PDFs downloaded successfully)

**Next Steps:**
1. 📋 **PDF Parsing** - Use pdfplumber/tabula-py to extract ranking tables
2. 📊 **Data Pipeline** - Convert PDFs → structured data (Parquet)
3. 📥 **Batch Download** - Download all historical years (1986-2026)
4. 🔄 **Weekly Updates** - Automated script to get latest rankings

**Command Reference:**
```bash
# Download single year
python scrapers/owgr_scraper.py --archive 2024 --download-pdfs data_files/owgr_pdfs

# Batch download (PowerShell)
2020..2025 | ForEach-Object { python scrapers/owgr_scraper.py --archive $_ --download-pdfs data_files/owgr_pdfs }

# Current rankings (live scraping)
python scrapers/owgr_scraper.py --fetch --playwright --limit 300
```
3. ✅ **Use Playwright** for current/live rankings when needed
4. Build a cached database of weekly rankings over time

**PDF Parsing Example:**
```python
import tabula
import pandas as pd

# Extract tables from OWGR PDF
pdf_file = "data_files/owgr_pdfs/ranking_2025_week01.pdf"
tables = tabula.read_pdf(pdf_file, pages="all", multiple_tables=True)

# First table usually contains the rankings
df = tables[0]
# Clean and standardize column names
df.columns = ['rank', 'last_week', 'name', 'country', 'avg_points', 'total_points', ...]
```

**Estimated effort:**
- PDF download implementation: ✅ **DONE** (see `owgr_scraper.py --archive`)
- PDF parsing implementation: 1-2 hours
- Playwright scraper for live data: ✅ **DONE** (see `owgr_scraper.py --fetch`)

**Files Created:**
1. `scrapers/owgr_scraper.py` - Complete scraper with PDF archive + live rankings support
2. `docs/owgr_scraping_analysis.md` - This analysis document

**Next Steps:**
1. Test PDF downloads: `python scrapers/owgr_scraper.py --archive 2025 --download-pdfs data_files/owgr_pdfs`
2. Implement PDF parser (separate script)
3. Build weekly ranking database (parquet files or SQLite)
