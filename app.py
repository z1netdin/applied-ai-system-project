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
    .stApp { background-color: #F7F4EE; }
    .block-container { max-width: 720px; padding-top: 4rem; padding-bottom: 4rem; }

    h1 {
        text-align: center;
        color: #2F3B38;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .subtitle {
        text-align: center;
        color: #6E7A74;
        font-size: 18px;
        line-height: 1.6;
        margin: 0.75rem auto 3rem auto;
        max-width: 520px;
    }
    .section-label {
        color: #8A8475;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-align: center;
        margin: 2.5rem 0 1rem 0;
    }

    .stButton > button {
        height: 88px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        border-radius: 14px;
        border: 1.5px solid #E3DED3;
        background-color: #FFFFFF;
        color: #2F3B38;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #C1694F;
        color: #C1694F;
        background-color: #FBEDE7;
    }
    .stButton > button[kind="primary"] {
        height: 64px;
        background-color: #C1694F;
        border: none;
        color: #FFFFFF;
        font-size: 19px;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #A8573F;
        color: #FFFFFF;
    }

    .pick-card {
        background-color: #FFFFFF;
        border: 1.5px solid #E3DED3;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 12px;
    }
    .pick-title { font-size: 19px; font-weight: 700; color: #2F3B38; }

    .footnote {
        text-align: center;
        color: #A3A89E;
        font-size: 13px;
        margin-top: 3.5rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("VibeFinder")
st.markdown(
    '<p class="subtitle">Music to help you feel better, matched to how you feel right now.</p>',
    unsafe_allow_html=True,
)

songs = load_songs("data/songs.csv")

SURPRISE_QUERY = "a good, widely enjoyable song for someone who isn't sure what they want to hear right now"

if st.button("Not sure? Let us pick for you", type="primary", use_container_width=True):
    st.session_state["active_query"] = SURPRISE_QUERY

st.markdown('<p class="section-label">Or choose what fits right now</p>', unsafe_allow_html=True)

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

    st.markdown('<p class="section-label">Your picks</p>', unsafe_allow_html=True)
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

        st.markdown('<p class="section-label">Your picks</p>', unsafe_allow_html=True)
        for rank, (song, _score, explanation) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {song['title']}** — {song['artist']}")
            reasons = [_clean_reason(r) for r in explanation.split("; ")]
            st.caption(", ".join(reasons))

st.markdown(
    '<p class="footnote">VibeFinder suggests music based on how you say you feel. '
    "It's a companion, not a substitute for professional care.</p>",
    unsafe_allow_html=True,
)
