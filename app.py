"""
Simple Streamlit web page for VibeFinder.

Same logic as src/main.py, just with a web page instead of a terminal.
Run with: streamlit run app.py
"""

import streamlit as st

from src.recommender import load_songs, recommend_songs, recommend_with_rag

st.set_page_config(page_title="VibeFinder")

st.title("VibeFinder")
st.caption(
    "A Module 3 recommender, extended with a RAG pipeline: "
    "retrieve real songs, then an AI explains them, checked by a guardrail."
)

songs = load_songs("data/songs.csv")

tab_rag, tab_structured = st.tabs(["Describe what you want (new, RAG)", "Fill in your taste (original)"])

with tab_rag:
    st.subheader("Click a request, or type your own")

    presets = {
        "Calm for studying": "something calm and acoustic for studying late at night",
        "Workout": "high energy pop for a workout",
        "Party": "fun danceable party song",
        "Rainy day": "sad song for a rainy day",
    }

    query = st.session_state.get("rag_query", "")

    cols = st.columns(len(presets))
    for col, (label, preset_query) in zip(cols, presets.items()):
        if col.button(label, use_container_width=True):
            query = preset_query
            st.session_state["rag_query"] = preset_query

    query = st.text_input("Or type your own request", value=query)

    if st.button("Find songs", key="rag_button", type="primary"):
        if not query.strip():
            st.warning("Click a request above, or type your own first.")
        else:
            with st.spinner("Searching the catalog, then asking Gemini..."):
                result = recommend_with_rag(query, songs, k=3)

            if result["source"] == "llm":
                st.success("AI answer from Gemini - the guardrail checked it, and it only talks about the songs found below.")
            else:
                st.info("Safe backup answer used instead (AI unavailable, or the guardrail blocked a bad answer).")

            st.markdown("**Songs retrieved by search:**")
            for song, score in result["retrieved"]:
                st.write(f"- {song['title']} by {song['artist']}  (match: {score:.3f})")

            st.markdown("**Explanation:**")
            st.write(result["explanation"])

with tab_structured:
    st.subheader("Fill in your taste")
    genres = sorted({s["genre"] for s in songs})
    moods = sorted({s["mood"] for s in songs})

    genre = st.selectbox("Favorite genre", genres)
    mood = st.selectbox("Favorite mood", moods)
    energy = st.slider("Target energy", 0.0, 1.0, 0.5, 0.05)
    likes_acoustic = st.checkbox("I like acoustic songs")

    if st.button("Get recommendations", key="structured_button"):
        user_prefs = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "likes_acoustic": likes_acoustic,
        }
        results = recommend_songs(user_prefs, songs, k=5)
        for rank, (song, score, explanation) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {song['title']}** by {song['artist']}  (Score: {score:.2f})")
            st.caption(explanation)
