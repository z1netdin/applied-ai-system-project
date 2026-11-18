from src.retrieval import SongIndex, load_song_notes


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


def test_notes_improve_matching_for_words_not_in_structured_fields():
    """Multi-source retrieval: a word like 'relaxing' isn't in any structured field
    (genre/mood/etc), but a hand-written note can use it, so search should still work."""
    songs = make_songs()  # "Chill Study Beats" (mood=chill) vs "Gym Hero" (mood=energetic)
    notes = {"Chill Study Beats": "A relaxing track good for unwinding after a long day."}

    without_notes = SongIndex(songs).search("relaxing", k=2)
    with_notes = SongIndex(songs, notes=notes).search("relaxing", k=2)

    assert all(score == 0.0 for _song, score in without_notes)
    assert with_notes[0][0]["title"] == "Chill Study Beats"
    assert with_notes[0][1] > 0.0


def test_load_song_notes_parses_markdown_headings(tmp_path):
    notes_file = tmp_path / "notes.md"
    notes_file.write_text(
        "## Song One\nA calm, relaxing tune.\n\n## Song Two\nAn intense, loud track.\n",
        encoding="utf-8",
    )

    notes = load_song_notes(str(notes_file))

    assert notes["Song One"] == "A calm, relaxing tune."
    assert notes["Song Two"] == "An intense, loud track."


def test_load_song_notes_returns_empty_dict_when_file_missing():
    assert load_song_notes("data/does_not_exist.md") == {}
