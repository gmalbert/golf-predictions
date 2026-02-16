"""
Update OWGR Rankings - Weekly Update Script

This script:
1. Fetches the latest OWGR PDF for the current week
2. Parses it
3. Adds player IDs
4. Appends to existing rankings
5. Rebuilds features
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.parse_owgr_pdfs_v2 import parse_owgr_pdf
from features.player_ids import PlayerRegistry, normalize_name


def get_current_week_year():
    """Get current ISO week and year."""
    now = datetime.now()
    iso_calendar = now.isocalendar()
    return iso_calendar.year, iso_calendar.week


def download_latest_owgr():
    """Download the most recent OWGR PDF if not already present."""
    print("\n" + "="*70)
    print("DOWNLOADING LATEST OWGR PDF")
    print("="*70)
    
    year, week = get_current_week_year()
    print(f"\n📅 Current: Year {year}, Week {week}")
    
    # Check if we already have data for this week
    existing_file = Path('data_files/owgr_rankings.parquet')
    if existing_file.exists():
        existing = pd.read_parquet(existing_file)
        latest_year = existing['source_year'].max()
        latest_week = existing[existing['source_year'] == latest_year]['source_week'].max()
        
        print(f"   Existing data: Year {latest_year}, Week {latest_week}")
        
        # If we're in the same year-week, don't download
        if latest_year == year and latest_week >= week:
            print(f"\n✅ Already have data for Week {week} - skipping download")
            return None
        
        # If current week is newer, download
        print(f"   → New week available: {week} > {latest_week}")
    
    pdf_dir = Path('data_files/owgr_pdfs')
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we already have PDF for the current week
    # OWGR PDFs typically follow pattern: owgr{week:02d}f{year}.pdf
    expected_patterns = [
        f'{year}_owgr{week:02d}f{year}.pdf',
        f'{year}_owgr{week:02d}*.pdf',
        f'owgr{week:02d}f{year}.pdf'
    ]
    
    existing_pdf = None
    for pattern in expected_patterns:
        matches = list(pdf_dir.glob(pattern))
        if matches:
            existing_pdf = matches[0]
            print(f"\n✅ Found existing PDF: {existing_pdf.name}")
            return existing_pdf
    
    # Download only the current week
    print(f"\n🌐 Downloading Week {week} PDF for {year}...")
    
    # Use Playwright to download just the latest ranking PDF
    from playwright.sync_api import sync_playwright
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            url = f"https://www.owgr.com/archive/{year}"
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            import time
            time.sleep(2)
            
            # Close any modals
            try:
                cookie_buttons = page.query_selector_all("button:has-text('Accept'), button:has-text('OK'), button:has-text('Close')")
                for btn in cookie_buttons[:1]:
                    try:
                        btn.click(timeout=2000)
                        time.sleep(1)
                        break
                    except:
                        pass
            except:
                pass
            
            # Select year in dropdown
            try:
                dropdown = page.query_selector("div.custom__select__control")
                if dropdown:
                    dropdown.click()
                    time.sleep(1)
                    year_option = page.query_selector(f"div[id*='react-select'][id*='option']:has-text('{year}')")
                    if year_option:
                        year_option.click()
                        time.sleep(2)
            except:
                pass
            
            # Wait for archive items
            try:
                page.wait_for_selector("a[href*='.pdf'], div.archivePageComponent", timeout=10000)
                time.sleep(1)
            except:
                pass
            
            # Get page HTML to see structure
            html = page.content()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all download buttons
            download_buttons = page.query_selector_all("div.archiveItemComponent_download__ju9MG")
            
            if not download_buttons:
                print("❌ No download buttons found")
                browser.close()
                return None
            
            print(f"   Found {len(download_buttons)} download buttons")
            
            # The page shows newest week first, and each week has 2 buttons (rankings + federation)
            # We want the first button (rankings) of the first week
            button = download_buttons[0]
            
            captured_url = []
            def handle_response(response):
                url = response.url
                if '.pdf' in url.lower():
                    # Capture any PDF response
                    captured_url.append(url)
                    print(f"   📄 Captured: {url.split('/')[-1][:50]}...")
            
            page.on("response", handle_response)
            
            print(f"   Clicking download button...")
            button.scroll_into_view_if_needed()
            time.sleep(0.5)
            button.click(force=True, timeout=5000)
            time.sleep(3)  # Wait longer for response
            
            page.remove_listener("response", handle_response)
            
            browser.close()
            
            if captured_url:
                # Download the PDF
                import requests
                import urllib.parse
                
                pdf_url = captured_url[0]
                
                # Skip federation rankings - only get main OWGR rankings
                if 'federation' in pdf_url.lower():
                    print("   ⚠️  First button was Federation rankings, not main OWGR")
                    print("      (This is expected - OWGR may not have published Week {week} yet)")
                    return None
                print(f"   📥 Downloading from: {pdf_url}")
                
                resp = requests.get(pdf_url, timeout=30)
                if resp.status_code == 200:
                    # Extract filename
                    parsed_url = urllib.parse.urlparse(pdf_url)
                    filename = parsed_url.path.split('/')[-1]
                    filename = urllib.parse.unquote(filename)
                    filename = filename.replace("/", "_").replace("\\", "_").replace(":", "-")
                    filename = f"{year}_{filename}"
                    
                    filepath = pdf_dir / filename
                    filepath.write_bytes(resp.content)
                    file_size = len(resp.content) / 1024
                    print(f"   ✓ Saved: {filename} ({file_size:.1f} KB)")
                    return filepath
                else:
                    print(f"   ✗ Download failed ({resp.status_code})")
                    return None
            else:
                print("   ❌ No PDF URL captured")
                return None
                
    except Exception as e:
        print(f"❌ Download error: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_owgr_rankings(new_pdf_path):
    """Parse new PDF and append to existing rankings."""
    print("\n" + "="*70)
    print("UPDATING OWGR RANKINGS")
    print("="*70)
    
    # Parse the new PDF
    print(f"\n📄 Parsing {new_pdf_path.name}...")
    new_data = parse_owgr_pdf(new_pdf_path)
    
    if new_data is None or new_data.empty:
        print("❌ No data extracted from PDF")
        return False
    
    print(f"   Extracted {len(new_data):,} ranking records")
    
    # Ensure consistent data types for numeric columns
    numeric_cols = [
        'rank_this_week', 'rank_last_week', 'rank_end_prev_year',
        'avg_points', 'total_points', 'events_played_divisor',
        'points_lost', 'points_gained', 'events_played_actual',
        'source_year', 'source_week', 'page_number'
    ]
    
    for col in numeric_cols:
        if col in new_data.columns:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    
    # Load existing rankings
    existing_file = Path('data_files/owgr_rankings.parquet')
    
    if existing_file.exists():
        print(f"\n📂 Loading existing rankings...")
        existing = pd.read_parquet(existing_file)
        print(f"   {len(existing):,} existing records")
        
        # Ensure existing data also has correct types
        for col in numeric_cols:
            if col in existing.columns:
                existing[col] = pd.to_numeric(existing[col], errors='coerce')
        
        # Check for duplicates (same year/week)
        new_year = new_data.iloc[0]['source_year']
        new_week = new_data.iloc[0]['source_week']
        
        duplicates = existing[
            (existing['source_year'] == new_year) & 
            (existing['source_week'] == new_week)
        ]
        
        if not duplicates.empty:
            print(f"\n⚠️  Found {len(duplicates)} existing records for {new_year} Week {new_week}")
            print("   Removing old data for this week...")
            existing = existing[
                ~((existing['source_year'] == new_year) & 
                  (existing['source_week'] == new_week))
            ]
        
        # Ensure new_data has same column order as existing
        new_data = new_data[existing.columns]
        
        # Append new data
        combined = pd.concat([existing, new_data], ignore_index=True)
    else:
        print(f"\n📝 Creating new rankings file...")
        combined = new_data
    
    # Save updated rankings
    print(f"\n💾 Saving {len(combined):,} total records...")
    combined.to_parquet(existing_file, index=False)
    
    print("✅ Rankings updated")
    return True


def add_player_ids_to_new_data():
    """Add player IDs to OWGR data."""
    print("\n" + "="*70)
    print("ADDING PLAYER IDS")
    print("="*70)
    
    owgr = pd.read_parquet('data_files/owgr_rankings.parquet')
    
    # Load or create registry
    registry = PlayerRegistry()
    
    # Add any new players
    unique_players = owgr[['player_name', 'country']].drop_duplicates()
    
    print(f"\n⚙️  Processing {len(unique_players):,} unique players...")
    
    for idx, row in unique_players.iterrows():
        registry.add_player(
            name=row['player_name'],
            country=row['country'],
            source='owgr',
            check_duplicates=False
        )
    
    registry.save()
    
    # Add player IDs to OWGR data
    owgr['normalized_name'] = owgr['player_name'].apply(normalize_name)
    registry_df = pd.read_parquet('data_files/player_registry.parquet')
    id_map = dict(zip(registry_df['normalized_name'], registry_df['player_id']))
    
    owgr['player_id'] = owgr['normalized_name'].map(id_map)
    owgr = owgr.drop(columns=['normalized_name'])
    
    # Save with IDs
    owgr.to_parquet('data_files/owgr_rankings_with_ids.parquet', index=False)
    
    matched = owgr['player_id'].notna().sum()
    print(f"✅ Added player_id to {matched:,}/{len(owgr):,} records")
    
    return True


def rebuild_features():
    """Rebuild OWGR features for ESPN data."""
    print("\n" + "="*70)
    print("REBUILDING OWGR FEATURES")
    print("="*70)
    
    try:
        # Import and run the feature builder directly
        sys.path.insert(0, str(Path(__file__).parent.parent / 'features'))
        from build_owgr_features import build_owgr_features
        
        df = build_owgr_features()
        
        if df is not None and not df.empty:
            print("✅ Features rebuilt successfully")
            return True
        else:
            print("❌ Feature rebuild returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ Feature rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the full weekly update process."""
    print("\n" + "="*70)
    print("🔄 WEEKLY OWGR UPDATE")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Download latest PDF
        latest_pdf = download_latest_owgr()
        if not latest_pdf:
            print("\n⚠️  No new PDF available - exiting")
            return
        
        # Step 2: Update rankings
        if not update_owgr_rankings(latest_pdf):
            print("\n❌ Failed to update rankings")
            return
        
        # Step 3: Add player IDs
        if not add_player_ids_to_new_data():
            print("\n❌ Failed to add player IDs")
            return
        
        # Step 4: Rebuild features
        if not rebuild_features():
            print("\n❌ Failed to rebuild features")
            return
        
        print("\n" + "="*70)
        print("✅ WEEKLY UPDATE COMPLETE!")
        print("="*70)
        
        # Show summary
        owgr = pd.read_parquet('data_files/owgr_rankings_with_ids.parquet')
        latest_year = owgr['source_year'].max()
        latest_week = owgr[owgr['source_year'] == latest_year]['source_week'].max()
        
        print(f"\nLatest data: Year {latest_year}, Week {latest_week}")
        print(f"Total rankings: {len(owgr):,}")
        print(f"Unique players: {owgr['player_id'].nunique():,}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
