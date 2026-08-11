"""Build student + admin interview reports."""

from __future__ import annotations

from typing import Any


def band(score: float) -> str:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "borderline"
    return "needs_work"


def recommendation_label(score: float) -> str:
    if score >= 80:
        return "recommend"
    if score >= 60:
        return "maybe"
    return "not_ready"


def build_report(state: dict[str, Any], turns: list[dict[str, Any]]) -> dict[str, Any]:
    conceptual = float(state.get("score_conceptual", 0))
    problem_solving = float(state.get("score_idea", 0))
    coding = float(state.get("score_coding", 0))
    explanation = float(state.get("score_explain", 0))
    communication = float(state.get("score_communication", 0))

    weights = {
        "conceptual": 0.2,
        "problem_solving": 0.2,
        "coding": 0.3,
        "explanation": 0.2,
        "communication": 0.1,
    }
    overall = (
        conceptual * weights["conceptual"]
        + problem_solving * weights["problem_solving"]
        + coding * weights["coding"]
        + explanation * weights["explanation"]
        + communication * weights["communication"]
    )

    strengths: list[str] = []
    gaps: list[str] = []
    dims = [
        ("Conceptual knowledge", conceptual),
        ("Problem-solving / approach", problem_solving),
        ("Coding correctness", coding),
        ("Explaining code under pressure", explanation),
        ("Communication clarity", communication),
    ]
    for name, val in sorted(dims, key=lambda x: -x[1]):
        if val >= 70 and len(strengths) < 3:
            strengths.append(f"{name} ({val:.0f})")
    for name, val in sorted(dims, key=lambda x: x[1]):
        if val < 70 and len(gaps) < 3:
            gaps.append(f"{name} ({val:.0f})")

    next_steps = []
    if coding < 70:
        next_steps.append("Practice array/hashmap problems on NexPractice with timed runs.")
    if explanation < 70:
        next_steps.append("After each solution, narrate every non-trivial block out loud.")
    if conceptual < 70:
        next_steps.append("Revise complexity, stacks, and database indexes with short written answers.")
    if not next_steps:
        next_steps.append("Take a harder mock with medium problems and stricter time.")

    return {
        "overall_score": round(overall, 1),
        "band": band(overall),
        "recommendation": recommendation_label(overall),
        "dimensions": {
            "conceptual": round(conceptual, 1),
            "problem_solving": round(problem_solving, 1),
            "coding": round(coding, 1),
            "explanation": round(explanation, 1),
            "communication": round(communication, 1),
        },
        "strengths": strengths or ["Showed up and completed the full loop"],
        "gaps": gaps or ["Keep sharpening edge-case discussion"],
        "next_steps": next_steps,
        "timeline": [
            {"stage": t.get("stage"), "role": t.get("role"), "preview": (t.get("content") or "")[:160]}
            for t in turns
            if t.get("role") in {"assistant", "student"}
        ][-40:],
        "problem_ids": state.get("used_problems", []),
        "flags": state.get("flags", []),
    }
