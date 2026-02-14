# Copilot Instructions

Purpose
-------
Short guidance for using GitHub Copilot and for contributors working on this repository.

Quick links
-----------
- Repository entry point: [predictions.py](predictions.py)
- Feature builders: [features/build_features.py](features/build_features.py)
- Scrapers: [scrapers/espn_golf.py](scrapers/espn_golf.py)
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

Editing guidelines for Copilot / contributors
-------------------------------------------
- Keep changes minimal and focused to solve the stated issue.
- Follow existing code style and naming conventions used in the repository.
- If you modify behavior, update the related docs under the `docs/` folder.
- When adding dependencies, update `requirements.txt` and explain why.
- Layout / UI guidelines:
- Avoid using `use_container_width`; use `width='stretch'` instead for Streamlit components and layout to ensure consistent cross-platform rendering.

Patch & commit workflow
-----------------------
- Make small, atomic commits with a short description prefix: `feat:`, `fix:`, `chore:`.
- Prefer descriptive commit bodies for non-trivial changes.

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
