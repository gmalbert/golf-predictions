# Scrapers

Web scraping utilities for Fairway Oracle.

## Features

- **Local caching** – Downloads are cached to `data_files/cache/` to avoid repeated requests
- **User agent rotation** – Randomly varies browser user agents to avoid detection
- **Polite delays** – 1.5-3.5 second random delays between requests
- **Error handling** – Graceful fallbacks when site structure changes

## Available Scrapers

### 1. ESPN Golf (`espn_golf.py`)

Scrapes PGA Tour tournament results from ESPN's JSON API.

**Coverage:** ~2001–present  
**Data quality:** ⭐⭐⭐⭐⭐ (official API, very reliable)

```bash
# Single year
python scrapers/espn_golf.py --year 2024

# Range of years
python scrapers/espn_golf.py --start 2020 --end 2024

# Test mode (no save)
python scrapers/espn_golf.py --year 2024 --no-save
```

**Output:** `data_files/espn_pga_{year}.parquet`

### 2. PGA Tour (`pga_tour_results.py`)

Scrapes tournament results directly from pgatour.com.

**Coverage:** 2004–present (varies by stat)  
**Data quality:** ⭐⭐⭐⭐ (official site, may change structure)

```bash
# Single year
python scrapers/pga_tour_results.py --year 2024

# Range of years
python scrapers/pga_tour_results.py --start 2020 --end 2024
```

**Output:** `data_files/pga_tour_{year}.parquet`

## Testing

Before scraping full datasets, test the scrapers:

```bash
# Run all tests
python scrapers/test_scrapers.py

# Clear cache and re-test
python scrapers/test_scrapers.py --clear-cache
```

This will:
1. Test ESPN API connectivity
2. Test PGA Tour site parsing
3. Verify local caching is working
4. Show sample data from each source

## Cache Management

HTML/JSON responses are cached in `data_files/cache/` as `.html` files (named by URL hash).

```python
# In your code
from shared_utils import clear_cache

# Clear all cached files
clear_cache()

# Or manually delete
rm -rf data_files/cache/*.html  # Unix
del data_files\cache\*.html     # Windows
```

**When to clear cache:**
- Site structure has changed and scraper needs updating
- Want to refresh data with latest tournament results
- Cache is taking too much disk space

## Data Output

All scrapers save to Parquet format in `data_files/`:

```python
import pandas as pd

# Load ESPN data
df = pd.read_parquet("data_files/espn_pga_2024.parquet")

print(df.columns)
# ['player_id', 'name', 'position', 'total_score', 'rounds', 
#  'tournament', 'date', 'year', ...]

print(f"Tournaments: {df['tournament'].nunique()}")
print(f"Players: {df['name'].nunique()}")
```

## Troubleshooting

### "No events found"
- Site structure may have changed
- Check if URL is still valid
- ESPN API may be temporarily down

### "Could not find __NEXT_DATA__"
- PGA Tour site structure changed
- Update the parsing logic in `pga_tour_results.py`
- Use ESPN as fallback data source

### "Rate limited / 429 error"
- Increase delay in `shared_utils.py` (currently 1.5-3.5s)
- Clear cache and try again later
- Use cached data for development

### Cache not working
- Check `data_files/cache/` directory exists
- Verify write permissions
- Look for errors in console output

## Best Practices

1. **Use cache during development** – Don't repeatedly hit live sites
2. **Test with single year first** – `--year 2024` before bulk scraping
3. **Scrape off-peak hours** – Be respectful of server load
4. **Keep user agents updated** – Rotate common browser strings
5. **Check robots.txt** – Respect site crawling rules

## Next Steps

After scraping data:

1. **Merge datasets** – Combine ESPN + PGA Tour for complete coverage
2. **Build features** – Run `features/build_features.py`
3. **Train models** – Run `models/baseline_logreg.py`

See the [roadmap docs](../docs/) for the complete implementation plan.
