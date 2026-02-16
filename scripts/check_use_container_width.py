#!/usr/bin/env python3
"""Pre-commit helper: prevent unnecessary width/use_container_width parameters."""
import sys
import re
from pathlib import Path

# Streamlit defaults to full-width, so these parameters are usually unnecessary
# and can cause version compatibility issues
BAD_PATTERNS = [
    r"st\.dataframe\([^)]*use_container_width\s*=",
    r"st\.dataframe\([^)]*width\s*=\s*['\"](?:stretch|content)['\"]"
]

exit_code = 0

# If no filenames passed, exit successfully (pre-commit always passes filenames)
if len(sys.argv) <= 1:
    sys.exit(0)

for fp in sys.argv[1:]:
    p = Path(fp)
    if not p.exists():
        continue
    if p.suffix.lower() != ".py":
        continue
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    
    for pattern in BAD_PATTERNS:
        if re.search(pattern, text):
            print(f"WARNING: Unnecessary width parameter in st.dataframe() found in {p}", file=sys.stderr)
            print("  Streamlit defaults to full-width, so parameters like use_container_width or width='stretch' are redundant.", file=sys.stderr)
            exit_code = 1
            break

if exit_code:
    print("\nSimply use: st.dataframe(df, hide_index=True)", file=sys.stderr)

sys.exit(exit_code)
