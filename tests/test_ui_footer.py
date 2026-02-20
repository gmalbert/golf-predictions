import re
from pathlib import Path


def test_ui_pages_include_footer():
    """Ensure any Streamlit-based UI page includes the Betting Oracle footer.

    The rule: any .py file that imports `streamlit` must reference
    `add_betting_oracle_footer` or import `footer` so the shared footer
    is rendered on UI pages.
    """
    repo_root = Path(__file__).resolve().parent.parent
    py_files = list(repo_root.rglob('*.py'))

    # Directories that are never project source code
    EXCLUDE_DIRS = {'venv', '.venv', 'env', '__pycache__', '.git', 'node_modules', 'site-packages'}

    ui_files = []
    for p in py_files:
        # Skip test files, the footer module, and third-party/virtual-env directories
        if p.match('**/tests/**') or p.name == 'footer.py':
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        text = p.read_text(encoding='utf-8', errors='ignore')
        if re.search(r"\bimport\s+streamlit\b|\bfrom\s+streamlit\b|\bst\.set_page_config\b", text):
            ui_files.append((p, text))

    assert ui_files, "No Streamlit UI pages found in repository."

    offenders = []
    for p, text in ui_files:
        if 'add_betting_oracle_footer' in text or 'from footer import' in text or 'footer.' in text:
            continue
        offenders.append(str(p.relative_to(repo_root)))

    assert not offenders, (
        "The following UI pages do not include the shared Betting Oracle footer:\n"
        + "\n".join(offenders)
        + "\n\nPlease add `from footer import add_betting_oracle_footer` and call `add_betting_oracle_footer()` at the end of the page."
    )
