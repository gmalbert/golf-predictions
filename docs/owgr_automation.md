# OWGR Weekly Update Automation

## Overview

Automated weekly updates of Official World Golf Ranking (OWGR) data via GitHub Actions.

## What It Does

Every Monday at 2 AM UTC, the workflow:

1. **📥 Downloads** the latest OWGR PDF from owgr.com/archive
2. **📊 Parses** the PDF to extract ranking data
3. **🔗 Links Players** by adding player IDs from the registry
4. **➕ Appends** new data to existing rankings (deduplicates if week already exists)
5. **🔧 Rebuilds Features** by regenerating OWGR features for ESPN tournament data
6. **💾 Commits** updated parquet files back to the repository

## Files Updated Weekly

- `data_files/owgr_rankings.parquet` - Raw OWGR rankings
- `data_files/owgr_rankings_with_ids.parquet` - Rankings with player IDs
- `data_files/espn_with_owgr_features.parquet` - ESPN data with OWGR features
- `data_files/player_registry.parquet` - Player ID registry
- `data_files/owgr_pdfs/YYYY_*.pdf` - Latest PDF (cached)

## Manual Trigger

To manually trigger an update:

1. Go to **Actions** tab in GitHub
2. Select **Update OWGR Rankings Weekly**
3. Click **Run workflow**
4. Select branch (usually `main`)
5. Click **Run workflow**

## Local Testing

Test the update script locally:

```bash
python tools/update_owgr_weekly.py
```

This will:
- Download the latest PDF
- Update all data files
- Rebuild features

**Note:** Requires Playwright for web scraping:
```bash
pip install playwright
playwright install chromium
```

## Schedule

- **Frequency:** Weekly (every Monday)
- **Time:** 2:00 AM UTC
- **Reason:** OWGR publishes new rankings on Sundays/Mondays

## Monitoring

### GitHub Actions Dashboard

Check workflow status at: `https://github.com/gmalbert/golf-predictions/actions`

### Commit Messages

Automated commits use format:
```
Update OWGR rankings - 2026 Week 7
```

### Workflow Summary

Each run creates a summary with:
- Latest data week/year
- Total ranking records
- Unique player count
- Update timestamp

## Troubleshooting

### No New PDF Found

If OWGR hasn't published for the current week yet, workflow exits gracefully.

### Playwright Browser Issues

The workflow installs Chromium automatically. If it fails:
```yaml
- name: Install Playwright browsers
  run: |
    playwright install chromium
    playwright install-deps chromium
```

### Parsing Errors

Check workflow logs for:
- PDF format changes
- Text extraction issues
- Player ID matching problems

Debug artifacts are uploaded on failure:
- Download from **Actions > Failed Run > Artifacts**

### Git Push Failures

If commits fail due to conflicts:
1. Pull latest changes locally
2. Re-run workflow manually
3. Check branch protection rules

## Data Quality

### Validation Checks

The update script validates:
- ✅ PDF downloaded successfully
- ✅ Data extracted (>0 rows)
- ✅ Player IDs assigned (100% coverage expected)
- ✅ No duplicate weeks (old data removed)
- ✅ Features rebuilt successfully

### Coverage

- **Current:** ~68.7% of ESPN tournament records have OWGR data
- **Target:** Increases as more weeks are added
- **Limitation:** Players outside top ~300 may not have OWGR data

## Cost Considerations

**GitHub Actions Free Tier:**
- 2,000 minutes/month for private repos
- Unlimited for public repos

**This Workflow:**
- ~5-10 minutes per run
- 4 runs/month (weekly)
- **Total:** ~40 minutes/month

Well within free tier limits. ✅

## Maintenance

### Annual PDF Format Changes

OWGR occasionally changes PDF layouts. If parsing breaks:

1. Check `scrapers/parse_owgr_pdfs_v2.py`
2. Run `tools/inspect_pdf.py` on latest PDF
3. Update regex patterns if needed
4. Test with `python tools/update_owgr_weekly.py`

### Player ID Conflicts

New players are automatically added to registry. If duplicates occur:

1. Check `data_files/player_registry.parquet`
2. Manually merge duplicates if needed
3. Update `features/player_ids.py` normalization rules

## Future Enhancements

Potential improvements:

- [ ] Email notifications on failure
- [ ] Slack/Discord webhook on completion
- [ ] Automatic backfill for missed weeks
- [ ] Data quality dashboard
- [ ] Historical trend analysis

## Related Documentation

- [OWGR Scraper](../scrapers/README.md)
- [Player ID System](../features/README.md)
- [Feature Engineering](../docs/04_models_and_features.md)

---

**Last Updated:** 2026-02-14  
**Owner:** @gmalbert
