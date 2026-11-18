"""
Reliability layer for the RAG pipeline: logging + a grounding guardrail.

The guardrail checks that the AI's generated explanation only talks about
songs that were actually retrieved. If the AI invents a song that was never
retrieved, or ignores the retrieved songs entirely, the guardrail blocks the
response and the caller falls back to a safe templated explanation instead.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "app.log"
LOG_PATH.parent.mkdir(exist_ok=True)

logger = logging.getLogger("vibefinder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)


def check_grounding(
    response_text: str, retrieved_songs: List[Dict], all_songs: List[Dict]
) -> Tuple[bool, str]:
    """
    Guardrail: the response must mention at least one retrieved song, and must
    not mention a catalog song that was NOT retrieved (that would mean the AI
    made up a recommendation instead of using the retrieved data).
    """
    retrieved_titles = {song["title"].lower() for song in retrieved_songs}
    other_titles = {
        song["title"].lower()
        for song in all_songs
        if song["title"].lower() not in retrieved_titles
    }
    text_lower = response_text.lower()

    mentions_other_song = any(title in text_lower for title in other_titles)
    if mentions_other_song:
        return False, "response mentions a song that was not retrieved"

    mentions_retrieved_song = any(title in text_lower for title in retrieved_titles)
    if not mentions_retrieved_song:
        return False, "response does not reference any retrieved song"

    return True, "grounded in retrieved songs"
