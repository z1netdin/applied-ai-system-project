# 🎵 VibeFinder: Applied AI Music Recommender

## Project Summary

**Base project:** This extends **VibeFinder 1.0**, a Module 3 mini-project. The original system is a content-based recommender: a user states their taste (favorite genre, favorite mood, target energy, and whether they like acoustic songs) through a structured profile, and the system scores every song in a small catalog based on how well it matches, returning the top 5 with plain-English reasons for each pick. It doesn't learn from other users or listening history — it only compares a song's own attributes to a one-time stated profile. That original logic still lives in `recommend_songs()` and runs as part of this app.

**What's new in this version (the Applied AI System extension):** a user can now type a **free-text request** (like "something calm and acoustic for studying late at night") instead of filling in a structured profile. The system:

1. **Retrieves** the closest-matching songs from the catalog for that request (`src/retrieval.py` — a small TF-IDF search index, no external search engine needed).
2. **Generates** a natural-language explanation using Google's Gemini model, but the model is only allowed to talk about the songs that were actually retrieved (`src/llm.py`).
3. **Checks (guardrail)** that the generated explanation didn't invent a song or ignore the retrieved data (`src/guardrails.py`). If it fails this check, or the AI call fails for any reason (no API key, network error, rate limit), the system **falls back** to a safe, deterministic templated explanation instead of crashing or showing a bad answer.

This is a Retrieval-Augmented Generation (RAG) feature: the AI's answer is grounded in data retrieved *before* it generates anything, and a reliability/guardrail layer double-checks the AI's output every time.

---

## How The System Works

Real recommenders like Spotify or YouTube mostly work in two ways: they look at what *other people* with similar taste liked (collaborative filtering), or they look at the *actual attributes* of the song or video itself, like genre and tempo, and match those to what you usually like (content-based filtering). For the collaborative side, platforms mostly watch what you *do*, not what you say: if you listen to a whole song, replay it, or click into it, that counts as a sign you liked it, and the system recommends more like it; if you skip it fast, that's a sign you didn't, so it recommends less like it. This happens automatically, from everyone's behavior, without anyone ever clicking a "like" button. Big platforms blend both approaches, but my version only does the content-based one, since I don't have a big pool of users generating listening behavior, just a small list of songs with known attributes and a one-time stated profile. My recommender takes a user's stated preferences (favorite genre, favorite mood, target energy, and whether they like acoustic songs) and compares them directly against each song's attributes, instead of learning from listening behavior over time like real systems do. I chose to prioritize mood and energy the most, since I think those two are what actually create a song's "vibe," while genre and acousticness matter too but are more about general taste than the specific feeling of a song.

Song features used: genre, mood, energy, acousticness. (tempo_bpm, valence, and danceability also exist on Song but aren't used, since UserProfile has no matching preference for them.)

UserProfile stores: favorite_genre, favorite_mood, target_energy, likes_acoustic.

To score a song, I add up points from four rules: +1.5 if the song's genre matches the user's favorite_genre, +2.0 if the mood matches favorite_mood, up to +2.0 based on how close the song's energy is to target_energy (closer energy means more points, using 2.0 * (1 - abs(song.energy - target_energy))), and +0.5/-0.5/0 depending on whether the song's acousticness lines up with likes_acoustic (bonus if it matches, penalty if it clearly conflicts, 0 if neutral). That gives each song a score out of about 6.0. I weighted mood a bit higher than genre because songs in the same genre can still feel very different (e.g. two pop songs, one happy and one intense), while songs in different genres can share the same mood, so mood felt like the stronger signal for matching vibe.

To pick which songs to recommend, I sort all the songs by that score from highest to lowest and return the top k (e.g. top 5). If two songs tie, I prefer the one with the smaller energy difference, since that's a more precise signal than the genre/mood matches.

Here's the data flow, start to finish:

```
Input: a UserProfile (favorite_genre, favorite_mood, target_energy, likes_acoustic)

Process: for every song in data/songs.csv
  - compare its genre, mood, energy, and acousticness to the UserProfile
  - add up the points from the scoring rules above
  - end up with a score (and a reason) for that one song

Output: sort every song by its score, highest first, and return the top k
```

**Algorithm Recipe (final):**

- Genre match: +1.5
- Mood match: +2.0
- Energy closeness: up to +2.0, using `2.0 * (1 - abs(song.energy - target_energy))`
- Acousticness alignment: +0.5 if it matches likes_acoustic, -0.5 if it conflicts, 0 if neutral
- Max possible score: 6.0

**Potential bias:** since I weighted mood higher than genre, this system might over-prioritize mood matches and surface songs that don't really fit a user's usual genre taste, just because the mood label happens to line up. It could also undervalue a song that's a near-perfect genre fit but only slightly off on mood. More generally, the whole system is biased toward however the genre/mood labels were assigned in the CSV, since it has no way to judge a song beyond those tags.

---

## New Feature: RAG Pipeline (Retrieve → Generate → Guardrail)

### Why this is RAG and not just "calling an LLM"

The AI model never sees the full song catalog and never answers from its own knowledge. It only ever sees the small set of songs that the retrieval step already picked out for this specific request. If the model tries to talk about a song outside that retrieved set, the guardrail blocks it. This is the difference between RAG and "ask an LLM a question": the retrieval step actively controls and limits what the model is allowed to say.

### Pipeline

Full system diagram (Mermaid source): [`diagrams/architecture.mmd`](diagrams/architecture.mmd).
It shows both input paths (structured profile and free-text query), the
retriever/generator/guardrail components, where the run gets logged, where a
human reviews that log, and which automated tests check which component.

```
Input: free-text request (e.g. "high energy pop for a workout")

1. Retrieve (src/retrieval.py)
   - TF-IDF search index built from every song's title/artist/genre/mood/energy/acousticness
   - Returns the top-k closest-matching songs for the request

2. Generate (src/llm.py)
   - Sends ONLY the retrieved songs' real attributes to Gemini (gemini-3.1-flash-lite)
   - Asks it to explain why those songs fit, using only the songs it was given
   - Any failure (no API key, network error, rate limit, server error) is caught
     and logged - it never crashes the app

3. Guardrail (src/guardrails.py)
   - Checks the generated text only mentions retrieved song titles
   - If it mentions a song that was NOT retrieved, or mentions no song at all -> reject
   - On reject (or if step 2 produced nothing), fall back to a safe templated
     explanation built directly from the retrieved songs' attributes

Output: an explanation the user can trust is grounded in the real catalog data,
plus a log entry recording what was retrieved and whether the guardrail passed
```

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for the AI-generated explanations) Add your Gemini API key:

   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   # then edit .env and paste your key:
   # GEMINI_API_KEY=your-key-here
   ```

   Get a free key at [Google AI Studio](https://aistudio.google.com/). **If you skip
   this step, the app still works** — the RAG pipeline automatically falls back to a
   templated explanation instead of calling the AI. This is a deliberate reliability
   feature, not a bug.

4. Run the app:

   ```bash
   python -m src.main
   ```

   Or try your own free-text request:

   ```bash
   python -m src.main --query "sad acoustic songs for a rainy day"
   ```

### Running Tests

```bash
pytest
```

11 tests cover the original scoring logic, the retrieval index, the guardrail, and
the RAG pipeline's fallback behavior (including simulating a missing/failing API key,
without needing a real network call).

---

## Sample Recommendation Output

Output of `python -m src.main` using the default profile (`genre=pop, mood=happy, energy=0.8`):

```
Loaded songs: 18

Top Recommendations
----------------------------------------
1. Sunrise City by Neon Echo  (Score: 5.46)
     - genre match (pop) (+1.5)
     - mood match (happy) (+2.0)
     - energy (0.82) close to your target (0.80) (+1.96)

2. Rooftop Lights by Indigo Parade  (Score: 3.92)
     - mood match (happy) (+2.0)
     - energy (0.76) close to your target (0.80) (+1.92)

3. Gym Hero by Max Pulse  (Score: 3.24)
     - genre match (pop) (+1.5)
     - energy (0.93) close to your target (0.80) (+1.74)

4. Carnival Lights by Luna Ray  (Score: 2.00)
     - energy (0.80) close to your target (0.80) (+2.00)

5. Night Drive Loop by Neon Echo  (Score: 1.90)
     - energy (0.75) close to your target (0.80) (+1.90)
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

---

## Sample RAG Output

Real output from `python -m src.main` (no `--query` given, so it runs 3 sample
requests), with a Gemini API key set:

```
Query: "something calm and acoustic for studying late at night"
(generated by: llm, guardrail passed: True)
Retrieved songs:
  - Night Drive Loop by Neon Echo (similarity: 0.098)
  - Sunrise City by Neon Echo (similarity: 0.000)
  - Midnight Coding by LoRoom (similarity: 0.000)
Explanation: For your late-night study session, "Midnight Coding" by LoRoom is the
perfect choice as it features a chill mood and high acousticness. While "Night Drive
Loop" by Neon Echo offers a moody atmosphere, its synthwave style makes it slightly
more energetic than the acoustic-leaning alternative.
----------------------------------------
Query: "high energy pop for a workout"
(generated by: llm, guardrail passed: True)
Retrieved songs:
  - Sunrise City by Neon Echo (similarity: 0.105)
  - Gym Hero by Max Pulse (similarity: 0.101)
  - Rooftop Lights by Indigo Parade (similarity: 0.098)
Explanation: These songs are perfect for a workout because they feature high energy
levels and upbeat pop sounds. "Gym Hero" provides the most intense rhythm for your
training, while "Sunrise City" and "Rooftop Lights" offer an uplifting, happy vibe
to keep you moving.
```

The same request, with **no API key set** (`GEMINI_API_KEY=` empty) - the system
degrades gracefully instead of crashing:

```
Query: "something calm for studying"
(generated by: fallback_template, guardrail passed: None)
Retrieved songs:
  - Sunrise City by Neon Echo (similarity: 0.000)
  - Midnight Coding by LoRoom (similarity: 0.000)
  - Storm Runner by Voltline (similarity: 0.000)
Explanation: Closest catalog matches for your request: "Sunrise City" by Neon Echo
(pop, happy mood); "Midnight Coding" by LoRoom (lofi, chill mood); "Storm Runner" by
Voltline (rock, intense mood).
```

---

## Guardrail Examples

These examples show the grounding guardrail (`src/guardrails.py::check_grounding`)
in action, including a forced failure case (simulated by feeding it a fake LLM
response) to prove it actually catches bad output rather than always passing.

**Example 1 - passes (AI stayed grounded in the retrieved songs)**

| | |
|---|---|
| Input (retrieved songs) | `"Chill Study Beats" by LoRoom` |
| AI response | `For studying, try "Chill Study Beats" by LoRoom - it is calm and acoustic.` |
| Guardrail result | ✅ PASS — `grounded in retrieved songs` |
| What happened | The response mentions a retrieved song and no other song, so it is used as-is. |

**Example 2 - fails (AI invented a song that was never retrieved)**

| | |
|---|---|
| Input (retrieved songs) | `"Chill Study Beats" by LoRoom` |
| AI response | `You should listen to "Unrelated Ballad" by Someone Else.` |
| Guardrail result | ❌ FAIL — `response mentions a song that was not retrieved` |
| What happened | The system **discards** this response and uses the safe templated fallback instead, so the user never sees an ungrounded recommendation. |

**Example 3 - fails (AI didn't mention any song)**

| | |
|---|---|
| Input (retrieved songs) | `"Chill Study Beats" by LoRoom` |
| AI response | `Here is a great song for you.` |
| Guardrail result | ❌ FAIL — `response does not reference any retrieved song` |
| What happened | Too vague to verify as grounded, so it is treated the same as a failure and falls back to the template. |

These three cases are also automated tests in `tests/test_guardrails.py`, so the
guardrail's behavior is checked on every run, not just by hand.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this



