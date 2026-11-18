"""
Evaluation harness for the RAG pipeline.

This is NOT pytest. It runs the real recommend_with_rag() pipeline against a
fixed list of test cases - normal queries AND edge cases (empty input,
gibberish input) - and prints a pass/fail table plus a one-line summary.
The point is to prove reliability with real, repeatable numbers instead of
just a demo that looks fine once.

Run it with: python -m src.evaluate
"""

from src.recommender import load_songs, recommend_with_rag

TEST_CASES = [
    {
        "name": "normal_calm_studying",
        "query": "something calm and acoustic for studying late at night",
        "criteria": "Returns >=1 retrieved song and a non-empty, grounded-or-safe explanation",
    },
    {
        "name": "normal_workout",
        "query": "high energy pop for a workout",
        "criteria": "Returns >=1 retrieved song and a non-empty, grounded-or-safe explanation",
    },
    {
        "name": "normal_happy",
        "query": "happy upbeat songs",
        "criteria": "Returns >=1 retrieved song and a non-empty, grounded-or-safe explanation",
    },
    {
        "name": "edge_empty_query",
        "query": "",
        "criteria": "Handles gracefully - no crash, still returns an explanation",
    },
    {
        "name": "edge_gibberish_query",
        "query": "xyzzy qqqq zzzrblx nonsense",
        "criteria": "Handles gracefully - no crash, no hallucinated unretrieved song",
    },
]


def run_case(case, songs):
    try:
        result = recommend_with_rag(case["query"], songs, k=3)
    except Exception as exc:  # noqa: BLE001 - a crash here IS the test failure
        return {"name": case["name"], "passed": False, "detail": f"CRASHED: {exc}"}

    has_songs = len(result["retrieved"]) > 0
    has_explanation = bool(result["explanation"].strip())
    # guardrail_passed is True (LLM grounded), or None (fallback used, no LLM call) -
    # both are safe outcomes. Only an explicit False-that-still-got-used would be bad,
    # and recommend_with_rag already swaps those out for the fallback template.
    safe_outcome = result["guardrail_passed"] in (True, None)

    passed = has_songs and has_explanation and safe_outcome
    detail = (
        f"source={result['source']} guardrail_passed={result['guardrail_passed']} "
        f"retrieved={len(result['retrieved'])}"
    )
    return {"name": case["name"], "passed": passed, "detail": detail}


def main() -> None:
    songs = load_songs("data/songs.csv")
    results = [run_case(case, songs) for case in TEST_CASES]

    print(f"{'Test Case':<24} {'Result':<6} Detail")
    print("-" * 90)
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{result['name']:<24} {status:<6} {result['detail']}")

    passed_count = sum(1 for r in results if r["passed"])
    print()
    print(f"Summary: {passed_count}/{len(results)} test cases passed")


if __name__ == "__main__":
    main()
