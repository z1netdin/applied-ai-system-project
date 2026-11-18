from src.retrieval import SongIndex


def make_songs():
    return [
        {
            "title": "Chill Study Beats",
            "artist": "LoRoom",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.3,
            "acousticness": 0.8,
        },
        {
            "title": "Gym Hero",
            "artist": "Max Pulse",
            "genre": "pop",
            "mood": "energetic",
            "energy": 0.95,
            "acousticness": 0.1,
        },
    ]


def test_search_ranks_matching_song_first():
    index = SongIndex(make_songs())
    results = index.search("chill lofi beats", k=2)

    assert results[0][0]["title"] == "Chill Study Beats"
    assert results[0][1] > results[1][1]


def test_search_returns_requested_k():
    index = SongIndex(make_songs())
    results = index.search("music", k=1)

    assert len(results) == 1


def test_search_is_deterministic():
    index = SongIndex(make_songs())
    first_run = index.search("energetic pop", k=2)
    second_run = index.search("energetic pop", k=2)

    assert [song["title"] for song, _ in first_run] == [song["title"] for song, _ in second_run]
