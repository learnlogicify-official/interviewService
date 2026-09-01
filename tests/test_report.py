"""Report scoring tests."""

from app.report import blend_explanation_score, idea_approach_explain_proxy


def test_idea_approach_counts_toward_explanation():
    state = {
        "score_explain": 0,
        "evidence": [
            {"stage": "idea", "dimension": "problem_solving", "score": 85},
            {"stage": "idea", "dimension": "problem_solving", "score": 70},
            {"stage": "idea", "dimension": "problem_solving", "score": 80},
        ],
    }
    proxy = idea_approach_explain_proxy(state)
    assert 75 <= proxy <= 83
    blended = blend_explanation_score(state)
    assert 60 <= blended <= 76


def test_live_explain_blends_with_idea():
    state = {
        "score_explain": 60,
        "evidence": [
            {"stage": "idea", "dimension": "problem_solving", "score": 80},
        ],
    }
    blended = blend_explanation_score(state)
    assert 65 <= blended <= 72


def test_coding_success_yields_explanation():
    from app.report import build_report

    state = {
        "score_idea": 75,
        "score_coding": 100,
        "score_explain": 0,
        "evidence": [
            {"stage": "idea", "dimension": "problem_solving", "score": 0, "note": "even divide odd multiply steps peak"},
            {"stage": "idea", "dimension": "problem_solving", "score": 0, "note": "global max equals input when n is one"},
            {"stage": "code", "dimension": "coding", "score": 100, "note": "All tests passed (10/10)"},
        ],
    }
    report = build_report(state, [{"role": "student", "stage": "idea", "content": "walk through collatz"}])
    assert report["dimensions"]["explanation"] >= 55
    assert report["dimensions"]["problem_solving"] >= 50
    idea_scores = [
        float(e["score"])
        for e in report["evidence"]
        if e.get("stage") == "idea" and e.get("dimension") == "problem_solving"
    ]
    assert any(s > 0 for s in idea_scores)


def test_explicit_explain_only():
    state = {"score_explain": 72, "evidence": []}
    assert blend_explanation_score(state) == 72.0
