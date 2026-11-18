from src import evaluate, recommender
from src.evaluate import TEST_CASES, run_case


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
        }
    ]


def test_run_case_passes_when_llm_is_grounded(monkeypatch):
    monkeypatch.setattr(
        recommender, "generate_explanation", lambda q, songs: f'"{songs[0]["title"]}" fits well.'
    )
    case = {"name": "sample", "query": "calm music", "criteria": "grounded"}

    result = run_case(case, make_songs())

    assert result["passed"] is True


def test_run_case_fails_on_crash(monkeypatch):
    def boom(query, songs, k=5):
        raise RuntimeError("boom")

    monkeypatch.setattr(evaluate, "recommend_with_rag", boom)
    case = {"name": "sample", "query": "calm music", "criteria": "no crash"}

    result = run_case(case, make_songs())

    assert result["passed"] is False
    assert "CRASHED" in result["detail"]


def test_all_predefined_cases_run_without_crashing(monkeypatch):
    """Smoke test: the harness's own test cases must run end-to-end without a real API call."""
    monkeypatch.setattr(recommender, "generate_explanation", lambda q, songs: None)

    for case in TEST_CASES:
        result = run_case(case, make_songs())
        assert "CRASHED" not in result["detail"]
