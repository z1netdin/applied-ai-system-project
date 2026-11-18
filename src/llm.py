"""
Thin wrapper around the Gemini API for the generation step of the RAG pipeline.

Design choice for reliability: this module never raises. If the API key is
missing, or the API call fails for any reason (network, rate limit, server
error), it logs the problem and returns None so the caller can fall back to
a safe templated explanation instead of crashing the app.
"""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

from src.guardrails import logger

load_dotenv()

MODEL_NAME = "gemini-3.1-flash-lite"

_client = None
_client_init_attempted = False


def _get_client():
    """Lazily creates the Gemini client. Returns None if no API key is set."""
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set - LLM generation disabled, using fallback mode")
        return None

    from google import genai

    _client = genai.Client(api_key=api_key)
    return _client


def generate_explanation(query: str, retrieved_songs: List[Dict]) -> Optional[str]:
    """
    Asks Gemini to explain why the retrieved songs fit the user's query,
    grounded only in the retrieved songs' real attributes. Returns None on
    any failure so the caller can use a safe fallback instead.
    """
    client = _get_client()
    if client is None:
        return None

    song_facts = "\n".join(
        f"- \"{song['title']}\" by {song['artist']}: genre={song['genre']}, "
        f"mood={song['mood']}, energy={song['energy']}, acousticness={song['acousticness']}"
        for song in retrieved_songs
    )
    prompt = (
        "You are a music recommendation assistant. A user asked for: "
        f"\"{query}\".\n\n"
        "Here are the only songs you are allowed to talk about (retrieved from "
        "the catalog as the closest matches):\n"
        f"{song_facts}\n\n"
        "For EACH song listed above, write exactly one line in this exact format:\n"
        "Song Title: one short sentence (10-20 words) explaining why it fits the request.\n\n"
        "Only reference songs from the list above, using their exact titles. "
        "Do not invent songs or attributes that are not listed. "
        "Output only those lines, one per song, nothing else before or after."
    )

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        text = (response.text or "").strip()
        if not text:
            logger.warning("LLM returned an empty response - using fallback")
            return None
        logger.info("LLM generation succeeded for query=%r", query)
        return text
    except Exception as exc:  # noqa: BLE001 - any failure should fall back, not crash
        logger.warning("LLM generation failed (%s) - using fallback", exc)
        return None
