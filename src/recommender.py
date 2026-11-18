import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.guardrails import check_grounding, logger
from src.llm import generate_explanation
from src.retrieval import SongIndex

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs.csv and returns a list of song dicts with numeric fields converted to int/float."""
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a song against user_prefs using the Algorithm Recipe, returning (score, reasons)."""
    score = 0.0
    reasons = []

    if song["genre"] == user_prefs.get("genre"):
        score += 1.5
        reasons.append(f"genre match ({song['genre']}) (+1.5)")

    if song["mood"] == user_prefs.get("mood"):
        score += 2.0
        reasons.append(f"mood match ({song['mood']}) (+2.0)")

    target_energy = user_prefs.get("energy")
    if target_energy is not None:
        energy_diff = abs(song["energy"] - target_energy)
        energy_points = 2.0 * (1 - energy_diff)
        score += energy_points
        reasons.append(
            f"energy ({song['energy']:.2f}) close to your target ({target_energy:.2f}) (+{energy_points:.2f})"
        )

    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        is_acoustic = song["acousticness"] >= 0.6
        if likes_acoustic and is_acoustic:
            score += 0.5
            reasons.append("matches your preference for acoustic songs (+0.5)")
        elif not likes_acoustic and is_acoustic:
            score -= 0.5
            reasons.append("more acoustic than you usually like (-0.5)")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every song, ranks by score, and returns the top k as (song, score, explanation)."""
    target_energy = user_prefs.get("energy")
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    def sort_key(item):
        song, score, _ = item
        energy_diff = abs(song["energy"] - target_energy) if target_energy is not None else 0.0
        return (-score, energy_diff)

    ranked = sorted(scored, key=sort_key)

    return [
        (song, score, "; ".join(reasons) if reasons else "no strong matches")
        for song, score, reasons in ranked[:k]
    ]


def _fallback_explanation(retrieved_songs: List[Dict]) -> str:
    """Safe, deterministic explanation used when the LLM is unavailable or fails the guardrail."""
    lines = [
        f"\"{song['title']}\" by {song['artist']} ({song['genre']}, {song['mood']} mood)"
        for song in retrieved_songs
    ]
    return "Closest catalog matches for your request: " + "; ".join(lines) + "."


def recommend_with_rag(query: str, songs: List[Dict], k: int = 5) -> Dict:
    """
    RAG pipeline for a free-text request, e.g. "something calm for studying":

    1. Retrieve: search the catalog for the k closest-matching songs (src/retrieval.py).
    2. Generate: ask the LLM to explain the picks using ONLY the retrieved songs (src/llm.py).
    3. Guardrail: check the explanation didn't invent songs or ignore the retrieved data
       (src/guardrails.py). If it fails, or the LLM is unavailable, fall back to a safe
       templated explanation so the system never crashes and always returns something useful.
    """
    index = SongIndex(songs)
    retrieved = index.search(query, k=k)
    retrieved_songs = [song for song, _score in retrieved]

    logger.info(
        "RAG query=%r retrieved=%r", query, [song["title"] for song in retrieved_songs]
    )

    explanation = generate_explanation(query, retrieved_songs)
    source = "llm"
    guardrail_passed = None
    guardrail_reason = "llm not used"

    if explanation is not None:
        guardrail_passed, guardrail_reason = check_grounding(explanation, retrieved_songs, songs)
        if not guardrail_passed:
            logger.warning("Guardrail blocked LLM response: %s", guardrail_reason)
            explanation = None

    if explanation is None:
        explanation = _fallback_explanation(retrieved_songs)
        source = "fallback_template"

    logger.info(
        "RAG result source=%s guardrail_passed=%s reason=%s",
        source,
        guardrail_passed,
        guardrail_reason,
    )

    return {
        "query": query,
        "retrieved": retrieved,
        "explanation": explanation,
        "source": source,
        "guardrail_passed": guardrail_passed,
        "guardrail_reason": guardrail_reason,
    }
