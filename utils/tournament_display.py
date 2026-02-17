"""Tournament display / sorting helpers used by the Streamlit UI and tests.

- format_tournament_display(name): returns a cleaned, readable label (Title Case with
  reasonable acronym preservation; strips leading year for Masters).
- tournament_sort_key(display_name): returns a lower-cased sort key that ignores
  leading articles (The/A/An).
"""
import re
from typing import Optional


def format_tournament_display(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str) or not name:
        return name
    display = name.strip()

    # Remove a leading 4-digit year only for Masters entries
    if 'masters' in display.lower():
        display = re.sub(r'^\s*\d{4}\s*[-–:]?\s*', '', display).strip()

    # Normalize leading article to Title form (fixes 'THE' -> 'The')
    display = re.sub(r'^(the|a|an)\b', lambda m: m.group(1).capitalize(), display, flags=re.I)

    # Title-case while preserving common acronyms (CJ, PGA, etc.) and tokens with dots (U.S.)
    ACRONYM_WHITELIST = {"CJ", "PGA", "LPGA", "RBC", "AT&T", "U.S.", "US"}
    parts = re.split(r'(\s+)', display)

    def _is_acronym(tok: str) -> bool:
        if not tok:
            return False
        # tokens containing dots (U.S.) are treated as acronyms
        if '.' in tok:
            return True
        # explicit whitelist (common tournament acronyms)
        if tok.upper() in ACRONYM_WHITELIST:
            return True
        # preserve very short all-caps tokens (2 letters, e.g. 'CJ')
        if tok.isalpha() and tok.isupper() and len(tok) == 2:
            return True
        return False

    def _fix_token(tok: str) -> str:
        if tok.isspace():
            return tok
        if _is_acronym(tok):
            return tok.upper()
        # Title-case hyphenated tokens and normal words; convert long ALL-CAPS words to Title form
        subtoks = re.split(r'(-)', tok)
        return ''.join(s.title() if s != '-' else s for s in subtoks)

    normalized = ''.join(_fix_token(p) for p in parts)
    return normalized


def tournament_sort_key(display_name: Optional[str]) -> str:
    """Return a case-insensitive sort key that ignores leading articles.

    Example: 'The Memorial Tournament' -> 'memorial tournament'
    """
    if not isinstance(display_name, str):
        return ''
    s = re.sub(r'^\s*(the|a|an)\s+', '', display_name, flags=re.I).strip()
    return s.lower()
