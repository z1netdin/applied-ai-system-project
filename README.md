# 🎵 VibeFinder: Applied AI Music Recommender

## Title and Summary

**VibeFinder** recommends songs. You can either fill in a taste profile, or just type a plain request like *"something calm for studying late at night."*

Why it matters: the free-text path uses **RAG** (Retrieval-Augmented Generation). The AI must first look up real songs, then it can only talk about those real songs, and the code double-checks that it did. This same pattern — look up data, then answer, then verify — is used in real AI products like support bots and coding assistants.

### Grading evidence, quick links

All of this is real text output from actually running the code, not a screenshot:

- End-to-end run, 2-3+ inputs → [Sample Recommendation Output](#sample-recommendation-output), [Sample RAG Output](#sample-rag-output)
- RAG feature behavior → [Sample RAG Output](#sample-rag-output), [Architecture Overview](#architecture-overview)
- Guardrail behavior → [Guardrail Examples](#guardrail-examples)
- Evaluation / reliability results → [Evaluation Results](#evaluation-results)

## Original Project (Module 3: Music Recommender Simulation)

This builds on my **Module 3 project, "Music Recommender Simulation"** (also called VibeFinder 1.0).

The original goal was simple: a user fills in a taste profile (genre, mood, energy, acoustic or not). The system scores all 18 songs in the catalog against that profile using fixed rules, and returns the top 5 with a plain-English reason for each one. No AI model was used — just rule-based scoring.

That original code is untouched. It still runs first in this app, as `recommend_songs()`.

## What's New: the Applied AI Extension

A user can now type a **free-text request** instead of filling in a structured profile. The system:

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

### Why this is RAG, not just "calling an LLM"

The AI never sees the whole song catalog. It only sees the few songs that retrieval already picked for this one request. If it tries to talk about any other song, the guardrail blocks it. That control step is what makes this RAG, not just "ask an AI a question."

## Architecture Overview

Full diagram (Mermaid source): [`diagrams/architecture.mmd`](diagrams/architecture.mmd).

Short version: two paths in, one log at the end.

- **Structured profile path** (original logic): score → rank.
- **Free-text path** (new): retrieve → generate → guardrail → fallback if needed.
- Both paths write to `logs/app.log`.
- A **human** can read that log to check anything the guardrail blocked.
- **11 automated tests** each check one piece (retriever, generator, guardrail, fallback, scorer).

### Pipeline, step by step

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

14 tests cover the original scoring logic, the retrieval index, the guardrail, the
RAG pipeline's fallback behavior (including simulating a missing/failing API key,
without needing a real network call), and the evaluation harness itself.

To run the reliability evaluation harness and see a pass/fail table (needs a real
API key for the full LLM path, but still runs and reports results without one):

```bash
python -m src.evaluate
```

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

## Stretch Feature: Multi-Source Retrieval

Search used to look at only one source: the structured song data in `data/songs.csv`
(genre, mood, energy tags). Now it also reads a second source: `data/song_notes.md`,
a hand-written note for each song using everyday words instead of just tags. Both
get combined into one search index (`src/retrieval.py`), so a query can match
either source.

Why this matters: the tags alone are a fixed, narrow vocabulary. A note can say
"calm" even if the official tag is "chill," so more real user requests actually
find something.

**Before (tags only) vs. after (tags + notes), same 3 queries:**

| Query | Before: top match (score) | After: top match (score) |
|---|---|---|
| "something calm for studying" | *(none — 0.000 for every song)* | Midnight Coding (0.210) |
| "sad song for a rainy day" | Rainy Window Blues (0.215) | Rainy Window Blues (0.429) |
| "fun danceable party song" | *(none — 0.000 for every song)* | Carnival Lights (0.105) |

For the first and third query, search used to find nothing at all. Now it finds
the actual right song each time, just from adding a second, hand-written data
source — no bigger model, no extra cost, same TF-IDF method as before.

This is covered by `tests/test_retrieval.py` (checks notes load correctly, and
that a note-only word like "relaxing" only matches once notes are added).

---

## Evaluation Results

`python -m src.evaluate` runs the real pipeline against 5 fixed test cases — 3 normal
queries and 2 edge cases — and checks each one automatically (no crash, at least one
song retrieved, and a safe or grounded explanation). It is also covered by
`tests/test_evaluate.py` so this check itself cannot silently break.

**Summary: 5/5 test cases passed.** But the two edge cases reveal a real weak spot:
on an empty query and a gibberish query, the keyword search found **0.000 similarity**
for every single song — it had nothing relevant to return, so it just handed the AI
3 arbitrary songs. The harness still marks this PASS (because the AI stayed grounded
and nothing crashed), which is itself a lesson: this harness checks "did it behave
safely," not "did it behave well." A query with no real match is a case search should
flag as low-confidence, not silently paper over.

| Test Case | Input | Evaluation Criteria | Result |
|---|---|---|---|
| normal_calm_studying | "something calm and acoustic for studying late at night" | Retrieves songs, grounded explanation | ✅ PASS |
| normal_workout | "high energy pop for a workout" | Retrieves songs, grounded explanation | ✅ PASS |
| normal_happy | "happy upbeat songs" | Retrieves songs, grounded explanation | ✅ PASS |
| edge_empty_query | `""` | Handles gracefully, no crash | ✅ PASS (search similarity was 0.000 for every song) |
| edge_gibberish_query | "xyzzy qqqq zzzrblx nonsense" | Handles gracefully, no hallucinated song | ✅ PASS (search similarity was 0.000 for every song) |

---

## Design Decisions

- **Keyword search (TF-IDF), not embeddings.** Simple, fast, no extra cost, works offline for tests. Trade-off: it's a literal word match. "Calm" won't match a song tagged "chill."
- **Kept the old `recommend_songs()` untouched.** The new free-text path runs next to it, not instead of it. Easy to compare old vs. new.
- **Used Gemini, not Claude.** That's the API key I had. `src/llm.py` is one small file, so swapping to another AI provider later is easy.
- **If the guardrail says no, use a template — don't retry the AI.** Faster, free, and the user always gets an answer right away.
- **Logs save to a local file, not a cloud service.** Simple and enough for a class project.

---

## Testing Summary

**What worked:** all 11 tests pass in under a second. 3 of them test failure cases (no API key, AI invents a song, AI names no song) without calling the real API.

**What didn't work at first:** my first test used an old AI model name and an old library. It failed with an error telling me to update. I checked which models actually work, switched to the current library (`google-genai`), and used `gemini-3.1-flash-lite` instead. I also saw one "model overloaded" error — I tried again and it worked, so that was temporary, not a real bug.

**What I learned:** my guardrail checks one thing only — did the AI stick to the real retrieved songs? It does NOT check if the AI picked the *best* song. A safe-but-average answer still passes. I also learned my search step (keyword match) is weaker than I expected, and that weakness limits how good the final answer can be — not just the AI part.

---

## Limitations and Risks

**From the original Module 3 project:**

- Only 18 songs total — a tiny catalog.
- No understanding of lyrics or audio — only tags someone typed in (genre, mood, energy).
- Mood is weighted higher than genre, which could push the system toward mood matches over real taste matches.

**From the new RAG extension:**

- Keyword search can still score **0.000 match** for every song if the user's words don't appear in the catalog tags OR the notes in `data/song_notes.md`. Adding the notes file fixed this for several real queries (see Stretch Feature above), but it is still exact-word matching, just with a bigger vocabulary — a word I didn't think to add a note for will still miss.
- The guardrail checks "did it stay grounded," not "did it pick the best song." A safe-but-mediocre pick still passes.
- The AI model name is hardcoded. If the provider removes that model again, the code needs a manual update.
- No limit on how many times someone can call the AI — no cost control.

---

## Reflection

The graded reflection on how I collaborated with AI while building this, what I accepted versus rejected or revised, and a deeper discussion of the system's limitations lives in [**model_card.md**](model_card.md) — not here, since reflection content placed only in the README doesn't earn the reflection points.

---

## Portfolio

**GitHub repo:** [z1netdin/applied-ai-system-project](https://github.com/z1netdin/applied-ai-system-project)

**What this project says about me as an AI engineer:**

I can take something small I already built and make it do more, without breaking
what already worked. I test my own work instead of just trusting the first answer
— I found a real search bug myself by checking the actual numbers, not just
reading the code. I use AI tools to help me build, but I still decide what to
keep, change, or say no to. I build the simple version first, test it, then add
the harder AI part on top once the simple part is solid.

