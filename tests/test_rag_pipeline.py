from src import recommender


def make_songs():
    return [
        {
            "id": 1,
            "title": "Chill Study Beats",
            "artist": "LoRoom",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.3,
            "tempo_bpm": 70,
            "valence": 0.5,
            "danceability": 0.3,
            "acousticness": 0.8,
        },
        {
            "id": 2,
            "title": "Energetic Workout Jam",
            "artist": "Max Pulse",
            "genre": "pop",
            "mood": "energetic",
            "energy": 0.95,
            "tempo_bpm": 150,
            "valence": 0.8,
            "danceability": 0.9,
            "acousticness": 0.1,
        },
    ]


def test_falls_back_to_template_when_llm_unavailable(monkeypatch):
    """Simulates a missing/failed API key: the system must still return a useful answer."""
    monkeypatch.setattr(recommender, "generate_explanation", lambda _query, _songs: None)

    result = recommender.recommend_with_rag("something calm for studying", make_songs(), k=1)

    assert result["source"] == "fallback_template"
    assert "Chill Study Beats" in result["explanation"]


def test_guardrail_rejects_hallucinated_llm_response(monkeypatch):
    """If the LLM invents a song that was never retrieved, the guardrail must block it."""
    monkeypatch.setattr(
        recommender,
        "generate_explanation",
        lambda _query, _songs: 'Try "Some Made Up Song" by Nobody - it is perfect for you.',
    )

    result = recommender.recommend_with_rag("something calm for studying", make_songs(), k=1)

    assert result["guardrail_passed"] is False
    assert result["source"] == "fallback_template"


def test_uses_llm_response_when_grounded(monkeypatch):
    monkeypatch.setattr(
        recommender,
        "generate_explanation",
        lambda query, songs: f'"{songs[0]["title"]}" fits your mood perfectly.',
    )

    result = recommender.recommend_with_rag("something calm for studying", make_songs(), k=1)

    assert result["source"] == "llm"
    assert result["guardrail_passed"] is True


def test_explanations_has_one_entry_per_retrieved_song(monkeypatch):
    monkeypatch.setattr(
        recommender,
        "generate_explanation",
        lambda query, songs: "Chill Study Beats: perfect for a quiet study session.",
    )

    result = recommender.recommend_with_rag("something calm for studying", make_songs(), k=1)

    assert "Chill Study Beats" in result["explanations"]
    assert "quiet study session" in result["explanations"]["Chill Study Beats"]


def test_explanations_falls_back_per_song_when_llm_unavailable(monkeypatch):
    monkeypatch.setattr(recommender, "generate_explanation", lambda _query, _songs: None)

    result = recommender.recommend_with_rag("something calm for studying", make_songs(), k=1)

    assert "Chill Study Beats" in result["explanations"]
    assert result["explanations"]["Chill Study Beats"].strip() != ""
