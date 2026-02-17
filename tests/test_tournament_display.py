from utils.tournament_display import format_tournament_display, tournament_sort_key


def test_format_preserves_acronyms_and_title_cases():
    assert format_tournament_display('THE CJ CUP Byron Nelson') == 'The CJ Cup Byron Nelson'
    assert format_tournament_display('u.s. open') == 'U.S. Open'
    assert 'CJ' in format_tournament_display('THE CJ CUP Byron Nelson')


def test_strip_leading_year_masters():
    assert format_tournament_display('2018 Masters Tournament') == 'Masters Tournament'


def test_sort_key_ignores_articles_and_case():
    inputs = [
        'THE CJ CUP Byron Nelson',
        'careerbuilder challenge',
        'CHARLES SCHWAB Challenge',
        'The Memorial Tournament',
    ]

    displays = [format_tournament_display(s) for s in inputs]
    sorted_displays = sorted(displays, key=tournament_sort_key)

    expected = [
        format_tournament_display('careerbuilder challenge'),
        format_tournament_display('CHARLES SCHWAB Challenge'),
        format_tournament_display('THE CJ CUP Byron Nelson'),
        format_tournament_display('The Memorial Tournament'),
    ]

    assert sorted_displays == expected
