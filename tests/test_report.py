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


def test_qa_only_hides_coding_dimensions_and_weights():
    from app.report import build_report

    state = {
        "include_coding": False,
        "score_conceptual": 83,
        "score_communication": 90,
        "score_idea": 0,
        "score_coding": 0,
        "score_explain": 0,
        "qa_scores": [80, 85, 84],
        "evidence": [
            {"stage": "qa", "dimension": "conceptual", "score": 83, "skill": "sql.joins"},
        ],
        "skill_graph": {
            "_touched": ["sql.joins"],
            "sql": {"label": "SQL", "children": {"joins": 0.61}},
        },
    }
    # Seed independence via evidence hint summary defaults.
    report = build_report(
        state,
        [
            {"role": "student", "stage": "qa", "content": "I am a final year student."},
            {"role": "assistant", "stage": "qa", "content": "Tell me about joins."},
            {"role": "student", "stage": "qa", "content": "Inner join returns matching rows."},
        ],
    )
    dims = report["dimensions"]
    assert "problem_solving" not in dims
    assert "coding" not in dims
    assert "explanation" not in dims
    assert dims["conceptual"] >= 70
    assert dims["communication"] >= 70
    # Without zero coding dims dragging the mean, overall stays near conceptual/comms.
    assert report["overall_score"] >= 70
    assert report.get("include_coding") is False
    gaps = " ".join(report.get("gaps") or []).lower()
    assert "coding" not in gaps
    assert "problem-solving" not in gaps
    assert "explaining code" not in gaps