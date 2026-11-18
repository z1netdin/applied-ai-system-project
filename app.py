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
    .block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 4rem; }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #E3DED3;
    }
    .topbar .brand { font-weight: 700; color: #2F3B38 !important; font-size: 16px; letter-spacing: 0.5px; }
    .topbar nav a {
        color: #6E7A74 !important;
        text-decoration: none;
        font-size: 15px;
        margin-left: 28px;
    }
    .topbar nav a:hover { color: #C1694F !important; }

    h1[data-testid="stHeading"], .stApp h1, [data-testid="stMarkdownContainer"] h1 {
        color: #2F3B38 !important;
        text-align: center;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    .subtitle {
        text-align: center !important;
        color: #6E7A74 !important;
        font-size: 18px;
        line-height: 1.6;
        margin: 0.75rem auto 1.5rem auto !important;
        max-width: 520px;
    }
    .how-it-works {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        margin: 1.5rem auto 2.5rem auto;
        flex-wrap: wrap;
    }
    .how-step { text-align: center; max-width: 180px; color: #6E7A74; font-size: 14px; }
    .how-step b { display: block; color: #2F3B38; font-size: 15px; margin-bottom: 2px; }

    .section-label {
        color: #8A8475 !important;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-align: center !important;
        margin: 2.5rem 0 1rem 0 !important;
    }

    .stButton > button {
        height: 88px;
        width: 100%;
        font-size: 18px;
        font-weight: 600;
        border-radius: 14px;
        border: 1.5px solid #D6CFC0;
        background-color: #FFFFFF;
        color: #2F3B38;
        box-shadow: 0 1px 4px rgba(47, 59, 56, 0.06);
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #C1694F;
        color: #C1694F;
        background-color: #FBEDE7;
        box-shadow: 0 4px 12px rgba(193, 105, 79, 0.15);
    }
    .stButton > button[kind="primary"] {
        height: 72px;
        background-color: #C1694F;
        border: none;
        color: #FFFFFF !important;
        font-size: 19px;
        box-shadow: 0 2px 8px rgba(193, 105, 79, 0.25);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #A8573F;
        color: #FFFFFF !important;
    }

    .pick-card {
        background-color: #FFFFFF;
        border: 1.5px solid #D6CFC0;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(47, 59, 56, 0.06);
    }
    .pick-title { font-size: 19px; font-weight: 700; color: #2F3B38; }

    .stTextInput label, .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #2F3B38 !important;
        font-weight: 600 !important;
    }
    .stTextInput input::placeholder { color: #9C9688 !important; opacity: 1 !important; }

    .about-box {
        background-color: #FFFFFF;
        border: 1.5px solid #E3DED3;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 3rem;
    }
    .about-box h4 { color: #2F3B38; margin-bottom: 8px; }
    .about-box p { color: #6E7A74; font-size: 15px; line-height: 1.6; margin: 0; }

    .footnote {
        text-align: center;
        color: #A3A89E;
        font-size: 13px;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
    </style>

    <div class="topbar">
        <div class="brand">VIBEFINDER</div>
        <nav>
            <a href="#how-it-works-anchor">How It Works</a>
            <a href="#about-anchor">About</a>
        </nav>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("VibeFinder")
st.markdown(
    '<p class="subtitle">Music to help you feel better, matched to how you feel right now.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div id="how-it-works-anchor" class="how-step" style="height:0; margin:0; padding:0;"></div>
    <div class="how-it-works">
        <div class="how-step"><b>1. Tell us how you feel</b>Pick a mood, or describe it yourself.</div>
        <div class="how-step"><b>2. We find a match</b>Real songs, chosen for that feeling.</div>
        <div class="how-step"><b>3. See your picks</b>With a short reason for each one.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

songs = load_songs("data/songs.csv")

SURPRISE_QUERY = "a good, widely enjoyable song for someone who isn't sure what they want to hear right now"

col_l, col_mid, col_r = st.columns([1, 3, 1])
with col_mid:
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
    custom_text = st.text_input("How are you feeling?", placeholder="e.g. overwhelmed and need to slow down")
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
    """
    <div id="about-anchor" style="height:0; margin:0; padding:0;"></div>
    <div class="about-box">
        <h4>About VibeFinder</h4>
        <p>VibeFinder matches your mood to real songs in our catalog, using search
        and a touch of AI to pick what fits. It's a companion for how you're
        feeling right now - not a substitute for professional care.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="footnote">If you are in crisis, please reach out to a mental health professional '
    "or a crisis line in your area.</p>",
    unsafe_allow_html=True,
)
