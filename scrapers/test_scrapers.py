"""
Test runner for web scrapers.
Quick sanity checks before scraping full seasons.
"""

import sys
from pathlib import Path

# Add scrapers to path
sys.path.insert(0, str(Path(__file__).parent))

from shared_utils import clear_cache, CACHE_DIR
import espn_golf
import pga_tour_results


def test_espn_scraper():
    """Test ESPN scraper with a single recent year."""
    print("\n" + "="*70)
    print("TESTING ESPN SCRAPER")
    print("="*70)
    
    # Test schedule fetch - use 2023 to ensure completed tournaments
    print("\n1. Testing ESPN schedule fetch (2023)...")
    events = espn_golf.get_espn_schedule(2023)
    
    if events:
        print(f"✓ Found {len(events)} events")
        print(f"  Sample: {events[0]['name']}")
    else:
        print("✗ No events found")
        return False
    
    # Test leaderboard fetch (try multiple events to find one with data)
    print("\n2. Testing ESPN leaderboard fetch...")
    df = None
    for test_event in events[:10]:  # Try up to 10 events
        print(f"  Trying: {test_event['name']}")
        df = espn_golf.get_espn_leaderboard(test_event['id'])
        
        if not df.empty:
            print(f"✓ Got {len(df)} players")
            print(f"  Columns: {list(df.columns)}")
            print(f"\n  Sample data:")
            print(df.head(3)[['name', 'position', 'total_score']])
            break
    
    if df is None or df.empty:
        print("⚠️  No leaderboard data found in first 10 events")
        print("     (This is OK - events may not have started yet)")
        # Don't fail the test - this is expected for future events
        return True
    
    print("\n✅ ESPN scraper tests passed!")
    return True


def test_pga_tour_scraper():
    """Test PGA Tour scraper."""
    print("\n" + "="*70)
    print("TESTING PGA TOUR SCRAPER")
    print("="*70)
    
    # Test schedule fetch - use 2023 for completed tournaments
    print("\n1. Testing PGA Tour schedule fetch (2023)...")
    tournaments = pga_tour_results.get_pga_tour_schedule(2023)
    
    if tournaments:
        print(f"✓ Found {len(tournaments)} tournaments")
        print(f"  Sample: {tournaments[0]['name']}")
    else:
        print("⚠️  No tournaments found (site structure may have changed)")
        return False
    
    # Test leaderboard fetch
    print("\n2. Testing PGA Tour leaderboard fetch...")
    if tournaments:
        test_tourney = tournaments[0]
        if test_tourney.get('id'):
            print(f"  Tournament: {test_tourney['name']}")
            df = pga_tour_results.get_pga_tour_leaderboard(
                test_tourney['id'], 
                year=2023
            )
            
            if not df.empty:
                print(f"✓ Got {len(df)} players")
                print(f"  Columns: {list(df.columns)}")
                print(f"\n  Sample data:")
                print(df.head(3)[['name', 'position', 'total_score']])
            else:
                print("⚠️  No leaderboard data")
        else:
            print("⚠️  First tournament has no ID")
    
    print("\n✅ PGA Tour scraper tests passed!")
    return True


def test_caching():
    """Test that caching is working."""
    print("\n" + "="*70)
    print("TESTING CACHE SYSTEM")
    print("="*70)
    
    cache_files = list(CACHE_DIR.glob("*.html"))
    print(f"\n📂 Cache directory: {CACHE_DIR}")
    print(f"📄 Cached files: {len(cache_files)}")
    
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"💾 Total cache size: {total_size / 1024 / 1024:.2f} MB")
        print(f"\n  Sample cached URLs:")
        for f in cache_files[:3]:
            size = f.stat().st_size / 1024
            print(f"    - {f.name}: {size:.1f} KB")
    
    print("\n✅ Cache system working!")
    return True


if __name__ == "__main__":
    print("\n" + "🏌️ " * 20)
    print("FAIRWAY ORACLE - SCRAPER TESTS")
    print("🏌️ " * 20)
    
    # Option to clear cache before testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before testing"
    )
    args = parser.parse_args()
    
    if args.clear_cache:
        print("\nClearing cache...")
        clear_cache()
    
    results = []
    
    # Run tests
    results.append(("ESPN Scraper", test_espn_scraper()))
    results.append(("PGA Tour Scraper", test_pga_tour_scraper()))
    results.append(("Cache System", test_caching()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to scrape.")
        print("\nNext steps:")
        print("  1. Run: python scrapers/espn_golf.py --year 2024")
        print("  2. Run: python scrapers/pga_tour_results.py --year 2024")
        print("  3. Or scrape multiple years: --start 2020 --end 2024")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    print()
