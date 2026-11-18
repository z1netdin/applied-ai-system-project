"""
Retrieval step for the RAG pipeline.

Builds a small search index over the song catalog so a free-text query
(e.g. "something calm for studying") can find the closest-matching songs
*before* any text is generated. This is plain TF-IDF + cosine similarity -
no extra ML libraries needed, so it is easy to read and runs instantly.

Multi-source retrieval: the index is built from TWO sources per song -
the structured attributes in data/songs.csv (genre, mood, energy, ...)
AND a hand-written free-text blurb from data/song_notes.md, which uses
everyday synonym words (e.g. "calm" for a song tagged "chill"). Combining
both sources catches queries the structured fields alone would miss.
"""

import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


def load_song_notes(path: str) -> Dict[str, str]:
    """
    Loads hand-written song descriptions from a markdown file.

    Expected format: one "## Song Title" heading per song, followed by a
    short free-text blurb. Returns {} if the file doesn't exist, so the
    retriever still works with just the structured catalog.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        text = f.read()
    notes = {}
    for title, blurb in re.findall(r"^##\s+(.+?)\s*\n(.+?)(?=\n##|\Z)", text, re.DOTALL | re.MULTILINE):
        notes[title.strip()] = blurb.strip()
    return notes


def _song_text(song: Dict, note: str = "") -> str:
    """Turns a song's attributes (and optional free-text note) into one text blob to search."""
    base = (
        f"{song['title']} {song['artist']} {song['genre']} {song['mood']} "
        f"energy {song['energy']} acousticness {song['acousticness']}"
    )
    return f"{base} {note}" if note else base


def _tokenize(text: str) -> List[str]:
    """Lowercases and splits into words, stripping punctuation (commas, periods, ...)
    so "calm," in a note matches a clean query word like "calm"."""
    return re.findall(r"[a-z0-9']+", text.lower())


class SongIndex:
    """A TF-IDF search index over a list of song dicts, optionally enriched with free-text notes."""

    def __init__(self, songs: List[Dict], notes: Optional[Dict[str, str]] = None):
        self.songs = songs
        notes = notes or {}
        self._docs = [
            _tokenize(_song_text(song, notes.get(song["title"], ""))) for song in songs
        ]
        self._doc_freq = Counter()
        for doc in self._docs:
            for term in set(doc):
                self._doc_freq[term] += 1
        self._n_docs = len(self._docs)
        self._doc_vectors = [self._vectorize(doc) for doc in self._docs]

    def _idf(self, term: str) -> float:
        return math.log((self._n_docs + 1) / (self._doc_freq.get(term, 0) + 1)) + 1

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        term_freq = Counter(tokens)
        return {term: count * self._idf(term) for term, count in term_freq.items()}

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        shared_terms = set(vec_a) & set(vec_b)
        dot_product = sum(vec_a[term] * vec_b[term] for term in shared_terms)
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def search(self, query: str, k: int = 5) -> List[Tuple[Dict, float]]:
        """Returns the top-k (song, similarity_score) pairs for a free-text query."""
        query_vector = self._vectorize(_tokenize(query))
        scored = [
            (song, self._cosine_similarity(query_vector, doc_vector))
            for song, doc_vector in zip(self.songs, self._doc_vectors)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
