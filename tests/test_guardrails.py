from src.guardrails import check_grounding


def make_catalog():
    return [
        {"title": "Chill Study Beats", "artist": "LoRoom"},
        {"title": "Gym Hero", "artist": "Max Pulse"},
        {"title": "Unrelated Ballad", "artist": "Someone Else"},
    ]


def test_grounded_response_passes():
    catalog = make_catalog()
    retrieved = [catalog[0]]
    response = 'For studying, try "Chill Study Beats" by LoRoom - it is calm and acoustic.'

    passed, reason = check_grounding(response, retrieved, catalog)

    assert passed is True
    assert "grounded" in reason


def test_response_mentioning_unretrieved_song_fails():
    catalog = make_catalog()
    retrieved = [catalog[0]]
    # The LLM hallucinated a song that was never retrieved.
    response = 'You should listen to "Unrelated Ballad" by Someone Else.'

    passed, reason = check_grounding(response, retrieved, catalog)

    assert passed is False
    assert "not retrieved" in reason


def test_response_mentioning_no_song_fails():
    catalog = make_catalog()
    retrieved = [catalog[0]]
    response = "Here is a great song for you."

    passed, reason = check_grounding(response, retrieved, catalog)

    assert passed is False
    assert "does not reference" in reason
