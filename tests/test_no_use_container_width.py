import subprocess
import re
from pathlib import Path


def test_no_unnecessary_width_params():
    """Warn if Python files use redundant width/use_container_width in st.dataframe."""
    # Only check tracked Python files
    files = subprocess.check_output(["git", "ls-files"]).decode().splitlines()
    offenders = []
    
    # Streamlit defaults to full-width, these parameters are usually redundant
    bad_patterns = [
        r"st\.dataframe\([^)]*use_container_width\s*=",
        r"st\.dataframe\([^)]*width\s*=\s*['\"](?:stretch|content)['\"]"
    ]
    
    for f in files:
        p = Path(f)
        if p.suffix.lower() != ".py":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        
        for pattern in bad_patterns:
            if re.search(pattern, text):
                offenders.append(f)
                break

    assert not offenders, (
        "Found unnecessary width parameters in st.dataframe() calls. "
        "Simply use st.dataframe(df, hide_index=True). Files:\n" + "\n".join(offenders)
    )
