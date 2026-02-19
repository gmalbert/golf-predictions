This folder contains planned follow-up 'issues' as markdown checklists for maintainers.

Files:
- `0001-sg-weather-tests.md` — unit tests for SG and weather enrichment
- `0002-model-v3-integration.md` — tasks to wire model-v3 into UI and tests

To convert a checklist item into a real GitHub Issue, run (example):

```bash
# create issue from file title + body
gh issue create --title "$(sed -n '1p' issues/0001-sg-weather-tests.md)" --body "$(sed -n '3,200p' issues/0001-sg-weather-tests.md)" --label "area:tests,priority:high"
```

Or copy/paste the checklist into a new issue in the GitHub web UI.