"""
VibeFinder web page.

Customer-facing copy only: no internal implementation details (model
names, retrieval mechanics, guardrail status, raw scores) are shown here.
That information exists for engineers in logs/app.log and the README.

Run with: streamlit run app.py
"""

import re

import streamlit as st

from src.recommender import load_songs, recommend_songs, recommend_with_rag

st.set_page_config(page_title="VibeFinder")

st.title("VibeFinder")
st.caption("Find your next favorite song.")

songs = load_songs("data/songs.csv")

tab_mood, tab_profile = st.tabs(["Describe Your Mood", "Build Your Profile"])


def _clean_reason(reason: str) -> str:
    """Drops the internal scoring math (e.g. "(+1.5)") from a reason string."""
    return re.sub(r"\s*\([+-]?\d+\.\d+\)", "", reason).strip()


with tab_mood:
    st.subheader("What are you in the mood for?")

    presets = {
        "Calm & Focused": "something calm and acoustic for studying late at night",
        "High Energy Workout": "high energy pop for a workout",
        "Party Vibes": "fun danceable party song",
        "Rainy Day": "sad song for a rainy day",
    }

    query = st.session_state.get("mood_query", "")

    cols = st.columns(len(presets))
    for col, (label, preset_query) in zip(cols, presets.items()):
        if col.button(label, use_container_width=True):
            query = preset_query
            st.session_state["mood_query"] = preset_query

    query = st.text_input("Or describe it yourself", value=query)

    if st.button("Find songs", key="mood_button", type="primary"):
        if not query.strip():
            st.warning("Choose a mood above, or describe your own first.")
        else:
            with st.spinner("Finding songs for you..."):
                result = recommend_with_rag(query, songs, k=3)

            st.markdown("#### Your picks")
            for song, _score in result["retrieved"]:
                st.markdown(f"**{song['title']}** — {song['artist']}")

            st.write(result["explanation"])

with tab_profile:
    st.subheader("Build your profile")
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    genre = st.selectbox("Favorite genre", genres)
    mood = st.selectbox("Favorite mood", moods)
    energy = st.slider("Target energy", 0.0, 1.0, 0.5, 0.05)
    likes_acoustic = st.checkbox("I like acoustic songs")

    if st.button("Get recommendations", key="profile_button", type="primary"):
        user_prefs = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "likes_acoustic": likes_acoustic,
        }
        results = recommend_songs(user_prefs, songs, k=5)

        st.markdown("#### Your picks")
        for rank, (song, _score, explanation) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {song['title']}** — {song['artist']}")
            reasons = [_clean_reason(r) for r in explanation.split("; ")]
            st.caption(", ".join(reasons))
