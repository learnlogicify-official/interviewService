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

    answered = bool(state.get("qa_scores")) or int(state.get("problems_solved_count", 0) or 0) > 0
    no_knowledge = int(state.get("no_knowledge_count", 0) or 0)
    qa_n = len(state.get("qa_scores") or [])
    substantive = answered and not (qa_n > 0 and no_knowledge >= max(1, int(qa_n * 0.7)))

    # Don't-know-heavy sessions must read as zeros, not soft partial credit.
    if not substantive:
        conceptual = 0.0 if no_knowledge >= qa_n and qa_n > 0 else min(conceptual, 5.0)
        communication = 0.0 if no_knowledge >= qa_n and qa_n > 0 else min(communication, 5.0)
        independence = 0.0
        hint_meta = dict(hint_meta)
        hint_meta["independence_score"] = 0.0
        hint_meta["independence_band"] = "hint_dependent"

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

    # Only surface skills that were actually tested this session (not untouched 50 defaults).
    touched = set(graph.get("_touched") or [])
    if not touched:
        for e in (graph.get("_evidence") or []):
            sk = str(e.get("skill") or "")
            if sk:
                touched.add(sk)
    skill_confidence: dict[str, Any] = {}
    for parent, node in graph.items():
        if str(parent).startswith("_"):
            continue
        kids_out = {}
        for child, val in (node.get("children") or {}).items():
            key = f"{parent}.{child}"
            if key in touched:
                kids_out[child] = float(val)
        if kids_out:
            skill_confidence[parent] = {
                "label": node.get("label", parent),
                "children": kids_out,
            }

    next_steps = []
    if not substantive:
        next_steps.append(
            "In the next mock, give a real attempt even if unsure — define the term, give one example, then trade-offs."
        )
        next_steps.append("Pick 2–3 weak topics from this session and write 5-sentence answers before your next interview.")
    if coding < 70:
        next_steps.append("Practice array/hashmap problems on NexPractice with timed runs until all tests pass.")
    if explanation < 70 and substantive:
        next_steps.append("After each solution, narrate every non-trivial block out loud.")
    if conceptual < 70:
        next_steps.append("Revise complexity, stacks, and database indexes with short written answers.")
    if independence < 60 and substantive:
        next_steps.append("Practice answering first without hints — state approach, then edge cases, then complexity.")
    if weak and touched:
        labels = ", ".join(w["label"] for w in weak[:2] if f"{w['parent']}.{w['child']}" in touched)
        if labels:
            next_steps.append(f"Drill weakest skills from this session: {labels}.")
    if not next_steps:
        next_steps.append("Take a harder mock with medium problems and stricter time.")

    titles = list(state.get("moodle_problem_titles") or [])
    if titles:
        next_steps.insert(0, "Problems in this session: " + ", ".join(titles[:3]) + ".")

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
        "skill_graph": skill_confidence,
        "skill_weakest": [w for w in weak if f"{w['parent']}.{w['child']}" in touched][:3],
        "skill_strongest": [w for w in strong if f"{w['parent']}.{w['child']}" in touched][:3],
        "skill_evidence": graph.get("_evidence") or [],
        "evidence": evidence_rows,
        "voice_metrics": state.get("voice_metrics") or {},
        "claims": state.get("claims") or [],
        "strengths": strengths if strengths else (
            ["Attempted the session"] if answered and not substantive
            else (["Showed up for the session"] if not answered
                  else ["No scored strengths yet — keep practicing"])
        ),
        "gaps": gaps if gaps else (
            ["Most replies were 'I don't know' — need substantive technical answers"]
            if answered and not substantive
            else (
                ["Complete at least one spoken answer and one passing submission"] if not answered
                else ["Keep sharpening edge-case discussion"]
            )
        ),
        "no_knowledge_count": no_knowledge,
        "substantive_answers": substantive,
        "next_steps": next_steps,
        "timeline": timeline[-50:],
        "problem_ids": state.get("used_problems", []),
        "flags": state.get("flags", []),
        "asked_question_ids": list(state.get("asked_question_ids") or []),
        "difficulty_ceiling_final": int(state.get("difficulty_ceiling", 0) or 0),
    }
