"""
VibeFinder web page.

Framed as a supportive music companion: pick how you feel, get songs right
away. Customer-facing copy only - no internal implementation details
(model names, retrieval mechanics, guardrail status, raw scores) are shown
here. That information exists for engineers in logs/app.log and the README.

Run with: streamlit run app.py
"""

import re

import streamlit as st

from src.recommender import load_songs, recommend_songs, recommend_with_rag

st.set_page_config(page_title="VibeFinder", layout="centered")

st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 3rem; }
    h1 { text-align: center; }
    .subtitle { text-align: center; color: #5B5B72; font-size: 19px; margin-bottom: 2.5rem; }
    .stButton > button {
        height: 92px;
        width: 100%;
        font-size: 19px;
        font-weight: 600;
        border-radius: 16px;
        border: 2px solid #E2E2EA;
        background-color: #FFFFFF;
        color: #1A1A2E;
    }
    .stButton > button:hover {
        border-color: #6C5CE7;
        color: #6C5CE7;
        background-color: #F5F3FF;
    }
    .stButton > button[kind="primary"] {
        height: 56px;
        background-color: #6C5CE7;
        border: none;
        color: white;
    }
    .pick-card {
        background-color: #FFFFFF;
        border: 2px solid #E2E2EA;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
    }
    .pick-title { font-size: 20px; font-weight: 700; color: #1A1A2E; margin-bottom: 4px; }
    .pick-why { font-size: 16px; color: #5B5B72; }
    .footnote { text-align: center; color: #9999AA; font-size: 14px; margin-top: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VibeFinder")
st.markdown('<p class="subtitle">How are you feeling right now?</p>', unsafe_allow_html=True)

songs = load_songs("data/songs.csv")

moods = {
    "Anxious": "calming songs to help ease anxiety and worry",
    "Feeling Low": "gentle comforting songs for a hard, heavy day",
    "Stressed": "relaxing songs to release stress and tension",
    "Need to Focus": "calm steady songs to help me focus on homework",
    "Can't Sleep": "slow soothing songs to help me wind down at night",
    "Need a Lift": "uplifting songs to help lift my mood",
}

if "active_query" not in st.session_state:
    st.session_state["active_query"] = None

mood_names = list(moods.keys())
row1, row2 = st.columns(3), st.columns(3)
for col, name in zip(row1, mood_names[:3]):
    if col.button(name, use_container_width=True):
        st.session_state["active_query"] = moods[name]
for col, name in zip(row2, mood_names[3:]):
    if col.button(name, use_container_width=True):
        st.session_state["active_query"] = moods[name]

with st.expander("Describe it in your own words instead"):
    custom_text = st.text_input("What's going on?", label_visibility="collapsed", placeholder="Tell us how you feel...")
    if st.button("Find songs", type="primary"):
        if custom_text.strip():
            st.session_state["active_query"] = custom_text

if st.session_state["active_query"]:
    with st.spinner("Finding songs for you..."):
        result = recommend_with_rag(st.session_state["active_query"], songs, k=3)

    st.markdown("#### Your picks")
    for song, _score in result["retrieved"]:
        st.markdown(
            f"""<div class="pick-card">
                <div class="pick-title">{song['title']} — {song['artist']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.write(result["explanation"])

with st.expander("Prefer to build a detailed taste profile instead?"):
    genres = sorted({s["genre"] for s in songs})
    profile_moods = sorted({s["mood"] for s in songs})

    genre = st.selectbox("Favorite genre", genres)
    mood = st.selectbox("Favorite mood", profile_moods)
    energy = st.slider("Target energy", 0.0, 1.0, 0.5, 0.05)
    likes_acoustic = st.checkbox("I like acoustic songs")

    if st.button("Get recommendations", key="profile_button"):
        user_prefs = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "likes_acoustic": likes_acoustic,
        }
        results = recommend_songs(user_prefs, songs, k=5)

        def _clean_reason(reason: str) -> str:
            return re.sub(r"\s*\([+-]?\d+\.\d+\)", "", reason).strip()

        st.markdown("#### Your picks")
        for rank, (song, _score, explanation) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {song['title']}** — {song['artist']}")
            reasons = [_clean_reason(r) for r in explanation.split("; ")]
            st.caption(", ".join(reasons))

st.markdown(
    '<p class="footnote">VibeFinder suggests music based on how you say you feel. '
    "It's a companion, not a substitute for professional care.</p>",
    unsafe_allow_html=True,
)
