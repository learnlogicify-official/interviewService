"""Build student + admin interview reports."""

from __future__ import annotations

from typing import Any

from app import evidence as evidence_ledger
from app import skills as skill_graph


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

    hint_meta = evidence_ledger.hint_summary(state)
    independence = float(hint_meta["independence_score"])
    depth = evidence_ledger.depth_metrics(state)

    weights = {
        "conceptual": 0.18,
        "problem_solving": 0.18,
        "coding": 0.28,
        "explanation": 0.16,
        "communication": 0.10,
        "independence": 0.10,
    }
    overall = (
        conceptual * weights["conceptual"]
        + problem_solving * weights["problem_solving"]
        + coding * weights["coding"]
        + explanation * weights["explanation"]
        + communication * weights["communication"]
        + independence * weights["independence"]
    )

    strengths: list[str] = []
    gaps: list[str] = []
    dims = [
        ("Conceptual knowledge", conceptual),
        ("Problem-solving / approach", problem_solving),
        ("Coding correctness", coding),
        ("Explaining code under pressure", explanation),
        ("Communication clarity", communication),
        ("Independence (low hint reliance)", independence),
    ]
    for name, val in sorted(dims, key=lambda x: -x[1]):
        if val >= 70 and len(strengths) < 3:
            strengths.append(f"{name} ({val:.0f})")
    for name, val in sorted(dims, key=lambda x: x[1]):
        if val < 70 and len(gaps) < 3:
            gaps.append(f"{name} ({val:.0f})")

    graph = state.get("skill_graph") or {}
    weak = skill_graph.weakest_skills(graph, 3)
    strong = skill_graph.strongest_skills(graph, 3)

    next_steps = []
    if coding < 70:
        next_steps.append("Practice array/hashmap problems on NexPractice with timed runs.")
    if explanation < 70:
        next_steps.append("After each solution, narrate every non-trivial block out loud.")
    if conceptual < 70:
        next_steps.append("Revise complexity, stacks, and database indexes with short written answers.")
    if independence < 60:
        next_steps.append("Practice answering first without hints — state approach, then edge cases, then complexity.")
    if weak:
        labels = ", ".join(w["label"] for w in weak[:2])
        next_steps.append(f"Drill weakest skills from this session: {labels}.")
    if not next_steps:
        next_steps.append("Take a harder mock with medium problems and stricter time.")

    evidence_rows = list(state.get("evidence") or [])[-24:]
    timeline = []
    for t in turns:
        if t.get("role") not in {"assistant", "student"}:
            continue
        timeline.append(
            {
                "stage": t.get("stage"),
                "role": t.get("role"),
                "preview": (t.get("content") or "")[:200],
            }
        )
    # Enrich timeline tail with scored evidence markers.
    for e in evidence_rows[-12:]:
        timeline.append(
            {
                "stage": e.get("stage"),
                "role": "evidence",
                "preview": (
                    f"{e.get('dimension')} {e.get('score')}/100 · "
                    f"{e.get('hint_label')} · {e.get('skill') or e.get('question_id')}"
                )[:200],
                "score": e.get("score"),
                "hint_level": e.get("hint_level"),
            }
        )

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
            "independence": round(independence, 1),
        },
        "independence": hint_meta,
        "depth": depth,
        "skill_graph": {
            k: v for k, v in graph.items() if not str(k).startswith("_")
        },
        "skill_weakest": weak,
        "skill_strongest": strong,
        "skill_evidence": graph.get("_evidence") or [],
        "evidence": evidence_rows,
        "voice_metrics": state.get("voice_metrics") or {},
        "claims": state.get("claims") or [],
        "strengths": strengths or ["Showed up and completed the full loop"],
        "gaps": gaps or ["Keep sharpening edge-case discussion"],
        "next_steps": next_steps,
        "timeline": timeline[-50:],
        "problem_ids": state.get("used_problems", []),
        "flags": state.get("flags", []),
        "asked_question_ids": list(state.get("asked_question_ids") or []),
        "difficulty_ceiling_final": int(state.get("difficulty_ceiling", 0) or 0),
    }
