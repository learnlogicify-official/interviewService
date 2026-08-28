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


def test_explicit_explain_only():
    state = {"score_explain": 72, "evidence": []}
    assert blend_explanation_score(state) == 72.0
