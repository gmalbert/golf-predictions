Title: Add unit tests for SG merges and weather enrichment

Checklist:
- [ ] Add unit tests for SG stat parsing and merge logic in `scrapers/pga_stats.py` and `features/build_extended_features.py`.
  - [ ] Test parsing of SG fields (sg_total, sg_putting, etc.) for edge cases and missing values.
  - [ ] Validate join keys and coverage after `player_id` normalization.
  - [ ] Add fixtures (small sample parquet/csv) for controlled test inputs.
- [ ] Add tests for weather enrichment logic (temperature, wind, precipitation) and parsing.
  - [ ] Verify correct units and default values when weather data missing.
  - [ ] Add a test that simulates a failed weather API response and confirms graceful degradation.
- [ ] Add CI job matrix entry (or extend existing tests) to run these tests on PRs.
- [ ] Document test data location and how to run tests locally: `pytest tests/test_pga_stats.py`.

Notes:
- Priority: High
- Owner: @you (assign in GitHub issue)
- ETA: 1 week

Related files: `scrapers/pga_stats.py`, `features/build_extended_features.py`, `tests/` directory

Suggested GitHub issue body (use with `gh issue create`):
> Add targeted unit tests to ensure SG stat parsing, SG merges, and weather enrichment are robust to malformed input and missing data. Include fixtures, edge-case checks, and CI coverage.