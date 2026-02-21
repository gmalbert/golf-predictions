# Copilot Instructions

Purpose
-------
Short guidance for using GitHub Copilot and for contributors working on this repository.

Quick links
-----------
- Repository entry point: [predictions.py](predictions.py)
- Feature builders: [features/build_features.py](features/build_features.py), [features/build_owgr_features.py](features/build_owgr_features.py)
- Models: [models/transformer_golfer.py](models/transformer_golfer.py), [models/course_embeddings.py](models/course_embeddings.py)
- Backtesting / bankroll: [evaluation/backtester.py](evaluation/backtester.py), [betting/bankroll.py](betting/bankroll.py)
- Live & alerts: [live/tournament_tracker.py](live/tournament_tracker.py), [monitoring/alerts.py](monitoring/alerts.py)
- Requirements: [requirements.txt](requirements.txt)

Environment setup (Windows PowerShell)
-----------------------------------
Run these commands to create and activate the virtual environment and install deps:

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the main script
-------------------
Run a quick smoke test:

```powershell
python predictions.py
```

- Useful command‑line helpers
---------------------------
```powershell
# train deep sequence model (Transformer/LSTM)
python models/transformer_golfer.py --epochs 10

# learn course embeddings and optionally cluster
python models/course_embeddings.py --cluster

# backtest historical performance + bankroll sim
python evaluation/backtester.py --start-year 2020 --kelly 0.25

# run live leaderboard tracker during an event
python live/tournament_tracker.py --event-id 401703511

# send a test Discord/email alert
python -c "from monitoring.alerts import send_discord_alert; send_discord_alert('test')"
```
Editing guidelines for Copilot / contributors
-------------------------------------------
- Keep changes minimal and focused to solve the stated issue.
- Follow existing code style and naming conventions used in the repository.
- If you modify behavior, update the related docs under the `docs/` folder.
- When adding dependencies, update `requirements.txt` and explain why.
- Layout / UI guidelines:
  - Streamlit defaults to full-width for tables; use `st.dataframe(df, hide_index=True)` for responsive tables.
  - Avoid `use_container_width` and avoid setting `width='stretch'` or `width='content'` in code — CI enforces this.

Debugging / data-workflow shortcuts
-----------------------------------
- Inspect ESPN API responses locally:
  - `python scrapers/test_espn_api.py` (quick JSON smoke test)
  - `python tools/inspect_espn_api.py` (detailed structure inspector)

- Run OWGR weekly update locally:
  - `python tools/update_owgr_weekly.py` (downloads PDF, parses, adds IDs, rebuilds OWGR features)

- Check OWGR feature coverage:
  - `python tools/check_owgr_features.py`

- Run the Streamlit UI:
  - `streamlit run predictions.py`

Pre-commit & tests
------------------
- Run `pre-commit run --all-files` before committing
- Run `python -m pytest -q` to execute the test suite

If you're unsure
---------------
Open a short issue describing the desired change before implementing large refactors.

Patch & commit workflow
-----------------------
- Make small, atomic commits with a short description prefix: `feat:`, `fix:`, `chore:`.
- Prefer descriptive commit bodies for non-trivial changes.
- **Branch separation:** keep topic branches isolated; do **not** merge or rebase them into main when pushing.  Instead, use `git push origin <branch>` and open a pull request when you’re ready to merge.  This makes it easier to track independent lines of work and roll back if needed.

Testing & verification
----------------------
- This repository does not include an automated test suite by default. After edits, run the modified module(s) locally and verify expected behavior.

Security & data
---------------
- Do not add secrets, API keys, or personal data to the repo. Use environment variables or local config files excluded via `.gitignore`.

If you're unsure
---------------
Open a short issue describing the desired change before implementing large refactors.

Thank you — keep changes small and well-documented.
