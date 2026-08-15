"""Evidence ledger + hint dependency (H0–H4) for interview scoring."""

from __future__ import annotations

from typing import Any


# H0: independent answer
# H1: clarification / rephrase of same question
# H2: soft probe (narrower follow-up)
# H3: directed nudge (deep probe without revealing)
# H4: near-solution (disallowed for interviewer; if somehow used, tanks independence)
HINT_LABELS = {
    0: "H0_independent",
    1: "H1_clarify",
    2: "H2_soft_probe",
    3: "H3_directed",
    4: "H4_near_reveal",
}


def clamp_hint(level: int) -> int:
    return max(0, min(4, int(level)))


def bump_hint(current: int, *, reason: str = "followup") -> int:
    """Raise hint level on weak/follow-up paths. Caps at H3 for normal probes."""
    cur = clamp_hint(current)
    if reason in {"followup", "probe_idea", "weak"}:
        return min(3, cur + 1)
    if reason == "deep_probe":
        return min(3, max(cur, 2) + 1) if cur < 3 else 3
    if reason == "reveal":
        return 4
    return cur


def record(
    state: dict[str, Any],
    *,
    stage: str,
    dimension: str,
    score: float,
    skill: str = "",
    question_id: str = "",
    hint_level: int = 0,
    note: str = "",
    source: str = "engine",
    elapsed: int = 0,
) -> None:
    bag = state.setdefault("evidence", [])
    entry = {
        "elapsed": int(elapsed or 0),
        "stage": stage,
        "dimension": dimension,
        "skill": skill or "",
        "question_id": question_id or "",
        "score": round(float(score), 1),
        "hint_level": clamp_hint(hint_level),
        "hint_label": HINT_LABELS[clamp_hint(hint_level)],
        "note": (note or "")[:240],
        "source": source,
    }
    bag.append(entry)
    state["evidence"] = bag[-80:]

    # Per-topic max hint used (for independence).
    deps = state.setdefault("hint_dependency", {})
    key = question_id or skill or dimension
    prev = int(deps.get(key, 0) or 0)
    deps[key] = max(prev, clamp_hint(hint_level))
    state["hint_dependency"] = deps


def independence_score(state: dict[str, Any]) -> float:
    """
    100 = answered everything at H0; lower as hints escalate.
    Weighted by evidence rows when present.
    """
    evidence = state.get("evidence") or []
    if evidence:
        penalties = []
        for e in evidence:
            h = clamp_hint(int(e.get("hint_level", 0) or 0))
            # H0=0, H1=12, H2=28, H3=48, H4=75 penalty points
            penalties.append({0: 0, 1: 12, 2: 28, 3: 48, 4: 75}.get(h, 20))
        avg_pen = sum(penalties) / len(penalties)
        return max(0.0, min(100.0, 100.0 - avg_pen))

    deps = state.get("hint_dependency") or {}
    if not deps:
        return 70.0  # neutral when no data
    levels = [clamp_hint(int(v)) for v in deps.values()]
    avg = sum(levels) / len(levels)
    return max(0.0, min(100.0, 100.0 - avg * 22.0))


def hint_summary(state: dict[str, Any]) -> dict[str, Any]:
    evidence = state.get("evidence") or []
    counts = {HINT_LABELS[i]: 0 for i in range(5)}
    for e in evidence:
        label = HINT_LABELS[clamp_hint(int(e.get("hint_level", 0) or 0))]
        counts[label] = counts.get(label, 0) + 1
    indep = independence_score(state)
    if indep >= 80:
        band = "high_independence"
    elif indep >= 55:
        band = "mixed_independence"
    else:
        band = "hint_dependent"
    return {
        "independence_score": round(indep, 1),
        "independence_band": band,
        "hint_counts": counts,
        "topics_touched": len(state.get("hint_dependency") or {}),
        "max_hint_seen": max([0] + [clamp_hint(int(v)) for v in (state.get("hint_dependency") or {}).values()]),
    }


def depth_metrics(state: dict[str, Any]) -> dict[str, Any]:
    """How often we stayed on a topic with follow-ups vs skimming."""
    evidence = state.get("evidence") or []
    qa = [e for e in evidence if e.get("stage") == "qa"]
    followups = sum(1 for e in qa if int(e.get("hint_level", 0) or 0) >= 1)
    topics = {e.get("question_id") or e.get("skill") for e in qa if e.get("question_id") or e.get("skill")}
    return {
        "qa_evidence_count": len(qa),
        "followup_turns": followups,
        "distinct_topics": len(topics),
        "avg_qa_score": round(sum(float(e.get("score", 0)) for e in qa) / max(1, len(qa)), 1) if qa else 0.0,
    }
