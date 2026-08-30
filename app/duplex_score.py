"""Duplex-native scoring for Realtime voice interviews.

Realtime invents spoken questions; the engine is an async scorer + stage FSM.
Scores are derived from the evidence ledger (what was actually judged), not from
peak floats alone — so fragment STT and empty engine replies cannot zero out a
session that clearly had substance.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app import evidence as evidence_ledger


# Minimum substance before a duplex utterance is scored as a full turn.
MIN_SCORE_WORDS = 12
MIN_SCORE_CHARS = 48

_YES_ONLY = re.compile(
    r"^\s*(yes|yeah|yep|yup|sure|confirm|please do|go ahead|end it|ok|okay|that's fine)\s*[.!]?\s*$",
    re.I,
)
_NO_ONLY = re.compile(
    r"^\s*(no|nope|continue|keep going|not yet|wait)\s*[.!]?\s*$",
    re.I,
)


def word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+", (text or "").lower()))


def is_confirm_yes(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    if _YES_ONLY.match(t):
        return True
    # Short affirmative that also mentions ending.
    if len(t.split()) <= 8 and re.search(r"\b(yes|yeah|yep|sure)\b", t):
        if re.search(r"\b(end|wrap|finish|stop|done|confirm)\b", t) or not re.search(
            r"\b(no|nope|not|continue|keep)\b", t
        ):
            return True
    return False


def is_confirm_no(text: str) -> bool:
    t = " ".join((text or "").lower().split())
    if _NO_ONLY.match(t):
        return True
    if len(t.split()) <= 8 and re.search(r"\b(no|nope|not yet|keep going|continue)\b", t):
        if not re.search(r"\b(yes|yeah|end it)\b", t):
            return True
    return False


def is_scoreable_utterance(text: str, *, stage: str = "qa") -> bool:
    """True when a duplex transcript is complete enough to score (not a mid-phrase chip)."""
    clean = " ".join((text or "").split())
    if not clean:
        return False
    if is_confirm_yes(clean) or is_confirm_no(clean):
        return True
    words = word_count(clean)
    if stage == "idea":
        return words >= 8 or len(clean) >= 32
    if stage in {"code", "explain"}:
        return words >= 6 or len(clean) >= 24
    return words >= MIN_SCORE_WORDS or len(clean) >= MIN_SCORE_CHARS


def heuristic_idea_score(text: str) -> float:
    """Floor score from approach language when the LLM returns 0 / misses."""
    blob = (text or "").lower()
    words = word_count(blob)
    if words < 8:
        return 15.0
    hits = 0
    for k in (
        "bfs", "dfs", "queue", "stack", "hash", "array", "graph", "dp",
        "dynamic", "complexity", "o(n)", "edge", "state", "energy",
        "shortest", "path", "loop", "sort", "pointer", "binary",
    ):
        if k in blob:
            hits += 1
    base = 35.0 + min(40.0, words * 0.9) + hits * 6.0
    return max(20.0, min(85.0, base))


def evidence_dimension_score(state: dict[str, Any], dimension: str) -> float:
    """Best*0.55 + avg*0.45 over evidence rows for a dimension — duplex-safe."""
    rows = [
        float(e.get("score", 0) or 0)
        for e in (state.get("evidence") or [])
        if str(e.get("dimension") or "") == dimension
    ]
    if not rows:
        return 0.0
    best = max(rows)
    avg = sum(rows) / len(rows)
    # Prefer peak so one strong answer isn't drowned by early weak fragments.
    return round(best * 0.55 + avg * 0.45, 1)


def recompute_scores(state: dict[str, Any]) -> dict[str, float]:
    """
    Rebuild canonical score_* floats from the evidence ledger.

    Peak state floats remain as floors (e.g. coding pass wrote 85) so we never
    discard a hard credit that evidence somehow missed.
    """
    conceptual_ev = evidence_dimension_score(state, "conceptual")
    idea_ev = evidence_dimension_score(state, "problem_solving")
    coding_ev = evidence_dimension_score(state, "coding")
    explain_ev = evidence_dimension_score(state, "explanation")

    qa_scores = [float(x) for x in (state.get("qa_scores") or []) if x is not None]
    qa_mean = (sum(qa_scores) / len(qa_scores)) if qa_scores else 0.0

    conceptual = max(float(state.get("score_conceptual") or 0), conceptual_ev, qa_mean)
    idea = max(float(state.get("score_idea") or 0), idea_ev)
    coding = max(float(state.get("score_coding") or 0), coding_ev)
    explain = max(float(state.get("score_explain") or 0), explain_ev)

    # Communication: blend voice metrics with answer length evidence.
    comm = float(state.get("score_communication") or 0)
    if conceptual > 0 or idea > 0 or coding > 0:
        # Floor communication from substance — duplex fragments used to leave this at 0.
        substantive_words = sum(
            word_count(str(e.get("note") or ""))
            for e in (state.get("evidence") or [])
            if e.get("dimension") in {"conceptual", "problem_solving", "explanation"}
        )
        comm_floor = min(75.0, 30.0 + substantive_words * 0.35)
        comm = max(comm, comm_floor * 0.5 + max(conceptual, idea) * 0.25)

    independence = evidence_ledger.independence_score(state)

    state["score_conceptual"] = round(conceptual, 1)
    state["score_idea"] = round(idea, 1)
    state["score_coding"] = round(coding, 1)
    state["score_explain"] = round(explain, 1)
    state["score_communication"] = round(min(100.0, comm), 1)

    return {
        "conceptual": state["score_conceptual"],
        "problem_solving": state["score_idea"],
        "coding": state["score_coding"],
        "explanation": state["score_explain"],
        "communication": state["score_communication"],
        "independence": round(independence, 1),
    }


def append_utterance_buffer(state: dict[str, Any], text: str, *, max_chars: int = 2500) -> str:
    """Merge a short duplex fragment into a pending buffer; return combined text."""
    prev = str(state.get("utterance_buffer") or "").strip()
    piece = " ".join((text or "").split())
    if not piece:
        return prev
    at = float(state.get("utterance_buffer_at") or 0)
    # Stale chips must not glue onto the next real answer minutes later.
    if prev and at and (time.time() - at) > 12.0:
        clear_utterance_buffer(state)
        prev = ""
    if not prev:
        combined = piece
    elif piece.lower() in prev.lower():
        combined = prev
    elif prev.lower() in piece.lower():
        combined = piece
    else:
        combined = (prev + " " + piece).strip()
    state["utterance_buffer"] = combined[:max_chars]
    state["utterance_buffer_at"] = time.time()
    return state["utterance_buffer"]


def clear_utterance_buffer(state: dict[str, Any]) -> None:
    state.pop("utterance_buffer", None)
    state.pop("utterance_buffer_at", None)


def _looks_like_continuation(prev: str, piece: str) -> bool:
    """True when piece is likely more of the same spoken answer, not a new turn."""
    a = " ".join((prev or "").lower().split())
    b = " ".join((piece or "").lower().split())
    if not a or not b:
        return False
    if b.startswith(a) or a.startswith(b):
        return True
    if a in b or b in a:
        return True
    # Shared tail/head overlap of a few words.
    a_words = a.split()
    b_words = b.split()
    if len(a_words) >= 3 and len(b_words) >= 3:
        if " ".join(a_words[-3:]) in b:
            return True
        if " ".join(b_words[:3]) in a:
            return True
    return False


def take_scoreable_text(state: dict[str, Any], text: str, *, stage: str) -> str | None:
    """
    Buffer weak fragments; return text only when ready to score.

    Returns None → caller should persist the student chip but skip LLM scoring.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return None
    if is_confirm_yes(clean) or is_confirm_no(clean):
        clear_utterance_buffer(state)
        return clean

    prev = str(state.get("utterance_buffer") or "").strip()
    at = float(state.get("utterance_buffer_at") or 0)
    if prev and at and (time.time() - at) > 12.0:
        clear_utterance_buffer(state)
        prev = ""

    # A fresh, already-complete answer must not inherit a sticky "I don't know" chip.
    if (
        prev
        and is_scoreable_utterance(clean, stage=stage)
        and not _looks_like_continuation(prev, clean)
    ):
        clear_utterance_buffer(state)
        return clean

    combined = append_utterance_buffer(state, clean)
    if is_scoreable_utterance(combined, stage=stage):
        clear_utterance_buffer(state)
        return combined
    # Keep buffering — do not score yet.
    return None
