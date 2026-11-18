"""
Retrieval step for the RAG pipeline.

Builds a small search index over the song catalog so a free-text query
(e.g. "something calm for studying") can find the closest-matching songs
*before* any text is generated. This is plain TF-IDF + cosine similarity -
no extra ML libraries needed, so it is easy to read and runs instantly.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple


def _song_text(song: Dict) -> str:
    """Turns a song's attributes into one text blob the index can search."""
    return (
        f"{song['title']} {song['artist']} {song['genre']} {song['mood']} "
        f"energy {song['energy']} acousticness {song['acousticness']}"
    )


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("-", " ").split()


class SongIndex:
    """A TF-IDF search index over a list of song dicts."""

    def __init__(self, songs: List[Dict]):
        self.songs = songs
        self._docs = [_tokenize(_song_text(song)) for song in songs]
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
