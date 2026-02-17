import re
from predictions import get_tournament_options
from utils.tournament_display import tournament_sort_key


def test_master_display_strips_leading_year_and_maps():
    options, mapping = get_tournament_options()
    # Ensure cleaned Masters display exists (no leading year in the label)
    assert any(opt.startswith('Masters Tournament (') for opt in options)

    # Pick a known Masters year and ensure mapping year is correct and underlying
    # dataset value still contains 'Masters' (we keep original dataset value)
    display = 'Masters Tournament (2018)'
    assert display in options
    underlying, year = mapping[display]
    assert year == 2018
    assert 'masters' in underlying.lower()


def test_cj_sorts_after_careerbuilder_and_charles_schwab():
    options, _ = get_tournament_options()
    # Find entries case-insensitively (display normalization varies)
    def find_opt(sub, year=None):
        for o in options:
            if sub.lower() in o.lower() and (year is None or f"({year})" in o):
                return o
        return None

    cb = find_opt('careerbuilder', 2018)
    cs = find_opt('charles schwab', 2025)
    cj = find_opt('cj', 2025)

    assert cb is not None
    assert cs is not None
    assert cj is not None

    assert options.index(cb) < options.index(cs) < options.index(cj)


def test_options_sorted_by_sort_key_then_year_desc():
    options, _ = get_tournament_options()
    parsed = []
    for opt in options:
        m = re.match(r'^(.*) \((\d{4})\)$', opt)
        assert m, f"Option not in expected format: {opt}"
        display = m.group(1)
        year = int(m.group(2))
        parsed.append((tournament_sort_key(display), -year, opt))

    assert parsed == sorted(parsed)
