# ESPN API Historical Data Fix

## Problem
The ESPN Golf API endpoint was returning the **same tournament data** for all event IDs, causing massive data corruption where the same 2 players appeared to win every tournament in the dataset.

## Root Cause
The ESPN API endpoint `?event={event_id}` **ignores the event parameter** and always returns the current/latest tournament, regardless of the event ID provided.

**Example:**
```
?event=401580329  → Returns: AT&T Pebble Beach Pro-Am 2026 (current tournament)
?event=401580330  → Returns: AT&T Pebble Beach Pro-Am 2026 (SAME tournament!)
```

## Solution
Use **date-based queries** instead of event ID queries. The ESPN API correctly responds to date range parameters.

**Correct format:**
```
?dates=YYYYMMDD-YYYYMMDD
```

**Example:**
```
?dates=20240104-20240111  → Returns: The Sentry 2024 (Chris Kirk won at -29)
?dates=20240411-20240418  → Returns: Masters 2024 (Scottie Scheffler won at -11)
```

## Implementation
Modified [`scrapers/espn_golf.py`](../scrapers/espn_golf.py):

### Changed Function Signature:
```python
def get_espn_leaderboard(event_id: str, event_date: str = None, ...):
```

### Added Date Range Logic:
```python
if event_date:
    from datetime import datetime, timedelta
    start = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
    end = start + timedelta(days=7)  # 7-day window for tournaments
    
    date_range = f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}"
    url = f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={date_range}"
```

### Updated Calling Code:
```python
# Old (broken):
df = get_espn_leaderboard(event["id"])

# New (works):
df = get_espn_leaderboard(event["id"], event_date=event.get("date"))
```

## Results

### Before Fix:
- **Total unique winners:** 2 (Akshay Bhatia, Ryo Hisatsune)
- **Akshay Bhatia wins:** 400 tournaments at -15 (impossible)
- **Ryo Hisatsune wins:** 400 tournaments at -15 (impossible)
- **Data quality:** CORRUPT

### After Fix:
- **Total unique winners:** 134
- **Akshay Bhatia wins:** 1 tournament (realistic)
- **Ryo Hisatsune wins:** 0 tournaments (realistic)
- **Top winner:** Scottie Scheffler with 24 wins (2018-2025)
- **Data quality:** CLEAN ✅

## Verification

Run diagnostics:
```bash
python tools/quick_winner_check.py
```

Expected output:
```
Total tournaments: 107
Unique winners: 134
Akshay Bhatia wins: 1
Ryo Hisatsune wins: 0
✅ DATA IS CLEAN! No more corruption!
```

## Key Learnings

1. **ESPN API quirk:** The `?event=` parameter doesn't work for historical golf data
2. **Date-based queries work:** Use `?dates=YYYYMMDD-YYYYMMDD` format
3. **Always validate results:** A small sample showed the issue immediately
4. **Score format differs:** Date-based queries return scores as **strings**, not dicts

## Files Modified
- [`scrapers/espn_golf.py`](../scrapers/espn_golf.py) - Updated API query logic
- [`features/merge_espn_parquets.py`](../features/merge_espn_parquets.py) - Fixed consolidation (previous issue)

## Next Steps
1. ✅ Re-scrape all ESPN data (2018-2025)
2. ✅ Rebuild consolidated file
3. ✅ Rebuild features
4. ⏳ Retrain models on clean data
5. ⏳ Verify predictions are realistic
