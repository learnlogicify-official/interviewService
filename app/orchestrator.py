"""Interview state machine orchestrator."""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app import evaluator
from app import evidence as evidence_ledger
from app import llm as llm_client
from app import question_graph
from app import skills as skill_graph
from app import voice_metrics
from app.config import get_settings
from app import duplex_score
from app.models import CodingSnapshotRow, SessionRow, TurnRow
from app.problems import get_problem, pick_problem, public_problem
from app.report import build_report, idea_approach_explain_proxy
from app.runner import evaluate_code


END_EXACT = {
    "done",
    "finish",
    "end",
    "stop",
    "quit",
    "wrap up",
    "that's all",
    "thats all",
    "i'm done",
    "im done",
    "end interview",
}
END_PHRASE_RE = re.compile(
    r"\b((i (want to|wanna|would like to|think we (can|should)) (end|stop|finish|wrap))|"
    r"(can|could|shall|should|may) we (please )?(end|stop|finish|wrap)|"
    r"(let'?s|lets) (just )?(end|stop|finish|wrap)|"
    r"(end|stop|finish) (the )?(interview|session|round)|"
    r"wrap (up|it up|this up)( the interview)?|"
    r"that'?s (it|all) (for|from) me|"
    r"i'?m done( here| with this| for today)?)\b",
    re.I,
)
YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm|please do|go ahead|end it|that's fine|ok)\b", re.I)
NO_RE = re.compile(r"\b(no|nope|continue|keep going|not yet|wait|don't|do not)\b", re.I)


def _is_end_intent(text: str) -> bool:
    t = " ".join((text or "").lower().split()).strip()
    if t in END_EXACT:
        return True
    # Spoken requests run long ("okay I think we can wrap up the interview now, thanks").
    if len(t.split()) <= 24 and END_PHRASE_RE.search(t):
        return True
    return False


def _is_yes(text: str) -> bool:
    return duplex_score.is_confirm_yes(text) or bool(YES_RE.search(text or ""))


def _is_no(text: str) -> bool:
    return duplex_score.is_confirm_no(text) or bool(NO_RE.search(text or ""))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _loads(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw or "") or default
    except Exception:
        return default


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


RESUME_TRACKS = {"resume_deep"}

RESUME_BRIEFING = (
    "RESUME-ONLY DEEP DIVE. No live coding, no editor, no LeetCode, no generic DSA bank. "
    "Ground every question in THIS candidate's resume: internships, projects, tech stack, "
    "ownership claims, metrics, coursework, and tools they listed. Be rigorous: architecture, "
    "trade-offs, failure modes, what THEY built vs the team, how they measured impact, how they "
    "debug. If they listed a skill, probe it with a scenario from THEIR project, not a textbook "
    "definition. Do not invent projects that are not on the resume. If the resume is thin, grill "
    "fundamentals of the tools they DID list. Never move_to_coding. next_action is followup or "
    "next_topic only. Score harshly for vague 'we used X' answers."
)


def _compose_profile_rules(
    *,
    style: str,
    difficulty: str,
    pace: str,
    question_mix: str,
    followup_depth: str,
    avoid_topics: str,
    include_coding: bool,
    topics: list[str] | None = None,
) -> str:
    """Turn structured custom-interviewer settings into briefing rules the LLM must follow."""
    style_rules = {
        "friendly": "Tone: warm, encouraging, still rigorous. Acknowledge briefly before probing.",
        "strict": "Tone: crisp and demanding. Push for precision; do not soft-pedal weak answers.",
        "brief": "Tone: concise. Keep acknowledgements under ~8 words; one sharp question.",
        "socratic": (
            "Tone: Socratic coach. Prefer guided questions that make the candidate reason aloud; "
            "rarely lecture. Dig into why/how before accepting surface answers."
        ),
        "supportive": (
            "Tone: supportive mentor. Reduce anxiety with calm pacing; still score honestly. "
            "Rephrase gently when answers are incomplete."
        ),
        "panel": (
            "Tone: panel-style hiring bar. Professional, structured, evidence-seeking. "
            "Ask for concrete examples, trade-offs, and measurable outcomes."
        ),
    }
    difficulty_rules = {
        "beginner": "Difficulty: beginner — fundamentals, definitions with examples, simple trade-offs.",
        "intermediate": "Difficulty: intermediate — applied scenarios, trade-offs, edge cases.",
        "advanced": "Difficulty: advanced — deeper design, failure modes, scale, and precision.",
    }
    pace_rules = {
        "relaxed": "Pace: relaxed — allow fuller answers; fewer topic jumps; patient follow-ups.",
        "standard": "Pace: standard campus screen — keep turns moving without rushing.",
        "brisk": "Pace: brisk — short follow-ups; move on after one clear rephrase if still weak.",
    }
    mix_rules = {
        "conceptual": "Question mix: spoken conceptual / trade-off. Avoid code-snippet formats unless briefing asks.",
        "mixed": "Question mix: mostly conceptual with occasional applied scenarios; snippets only if useful.",
        "behavioral": (
            "Question mix: behavioral + ownership (STAR-style). Ask for situation, action THEY took, result. "
            "Still technical when the claim is technical."
        ),
        "system_design": (
            "Question mix: lightweight system / architecture design for early-career. "
            "Components, data flow, bottlenecks — keep scope small for voice."
        ),
    }
    depth_rules = {
        "light": "Follow-up depth: light — at most one rephrase, then move topic.",
        "moderate": "Follow-up depth: moderate — up to two probes on the same competency.",
        "deep": "Follow-up depth: deep — up to three probes before changing topic.",
    }
    parts = [
        "CUSTOM INTERVIEWER RULES (MUST FOLLOW — same rules for every turn):",
        style_rules.get(style, style_rules["friendly"]),
        difficulty_rules.get(difficulty, difficulty_rules["intermediate"]),
        pace_rules.get(pace, pace_rules["standard"]),
        mix_rules.get(question_mix, mix_rules["conceptual"]),
        depth_rules.get(followup_depth, depth_rules["moderate"]),
        "Coding editor: " + ("ENABLED later in the session." if include_coding else "DISABLED — spoken Q&A only."),
    ]
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()][:12]
    if topic_list:
        numbered = " → ".join(f"{i}. {t}" for i, t in enumerate(topic_list, 1))
        parts.append("AGENDA (cover in this exact order): " + numbered + ".")
        parts.append(
            "Coverage rule: the FIRST spoken question after the greeting MUST be item 1. "
            "Do not skip ahead. Work through each listed area before coding. "
            "Ask at least one concrete question in each item when time allows."
        )
    if avoid_topics:
        parts.append("Do NOT ask about: " + avoid_topics.strip()[:400] + ".")
    parts.append(
        "Invent fresh questions aligned to these rules. Do not default to generic hash-map / Big-O openers "
        "unless topics or briefing explicitly include them."
    )
    if include_coding:
        parts.append(
            "Do not set next_action=move_to_coding until the spoken technical round has covered "
            "most focus topics or the engine closes Q&A on time. Prefer next_topic / followup."
        )
    return "\n".join(parts)


def _opening_already_greets(text: str, name: str) -> bool:
    """True if the LLM opening already has a greeting or self-intro (avoid double intro)."""
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"(?i)\b(hi|hey|hello|welcome|good (morning|afternoon|evening))\b", t):
        return True
    n = (name or "").strip()
    if n and n.lower() in t.lower():
        return True
    if re.search(r"(?i)\b(i'?m|i am|this is)\b", t[:120]):
        return True
    return False


def _is_human_resume_line(ln: str) -> bool:
    s = (ln or "").strip()
    if len(s) < 8:
        return False
    letters = len(re.findall(r"[A-Za-z]", s))
    if letters < 6:
        return False
    if letters / max(1, len(s)) < 0.42:
        return False
    if re.search(r"[ÿþßŠ¢ãÞµ«»¤¦§œžÐ]{2,}", s) and letters < 12:
        return False
    return True


def _normalize_resume(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    lines = [ln.strip() for ln in t.splitlines() if _is_human_resume_line(ln)]
    t = "\n".join(lines).strip()
    return t[:16000]


def _next_resume_item(state: dict[str, Any]) -> dict[str, Any] | None:
    plan = list(state.get("resume_plan") or [])
    idx = int(state.get("resume_plan_index") or 0)
    if 0 <= idx < len(plan) and isinstance(plan[idx], dict):
        return plan[idx]
    return None


def _is_resume_track(role_track: str = "", state: dict[str, Any] | None = None) -> bool:
    track = (role_track or "").strip()
    if state:
        track = track or str(state.get("role_track") or "")
        if state.get("resume_only"):
            return True
    return track in RESUME_TRACKS


def _norm_q(text: str) -> str:
    t = re.sub(r"```[\s\S]*?```", " ", text or "")
    t = re.sub(r"[^a-z0-9\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def _is_similar_q(a: str, b: str) -> bool:
    na, nb = _norm_q(a), _norm_q(b)
    if not na or not nb:
        return False
    if na == nb or na[:140] == nb[:140]:
        return True
    if len(na) > 48 and na[:72] in nb:
        return True
    if len(nb) > 48 and nb[:72] in na:
        return True
    return False


def _last_assistant_texts(db: Session, session_id: str, limit: int = 8) -> list[str]:
    rows = (
        db.query(TurnRow)
        .filter(TurnRow.session_id == session_id, TurnRow.role == "assistant")
        .order_by(TurnRow.seq.desc())
        .limit(limit)
        .all()
    )
    return [str(r.content or "") for r in rows]


def _dedupe_against_history(db: Session, session_id: str, reply: str) -> str:
    """Drop a reply that restates a recent interviewer question."""
    text = (reply or "").strip()
    if not text:
        return text
    # Intentional re-ask / rephrase turns must be allowed (voice STT often needs this).
    if re.search(
        r"(?i)\b("
        r"didn'?t catch|did not catch|not sure i (heard|follow|got)|"
        r"came through (unclear|incomplete)|let me (ask|rephrase|put that)|"
        r"ask (that|this) again|same question|another way|more simply|"
        r"doesn'?t (quite )?answer|not (really )?related|off[- ]?topic"
        r")\b",
        text,
    ):
        return text
    for prev in _last_assistant_texts(db, session_id):
        if _is_similar_q(text, prev):
            return ""
    return text


def _current_qa_stem(state: dict[str, Any], *, hint_bump: int = 1) -> str:
    """Best spoken stem for the active Q&A node / resume item."""
    hint = max(0, min(3, int(state.get("current_hint_level", 0) or 0) + hint_bump))
    followup_index = int(state.get("followup_index", 0) or 0)
    qid = str(state.get("current_question_id") or "")
    node = question_graph.get_node(qid) if qid else None
    if node:
        return question_graph.spoken_prompt(
            node,
            hint_level=max(1, hint),
            followup_index=followup_index,
        )
    nxt = _next_resume_item(state)
    if nxt and nxt.get("question"):
        return str(nxt["question"]).strip()
    spoken = str(state.get("spoken_now") or state.get("last_question") or "").strip()
    return spoken


def _reask_current_question(
    state: dict[str, Any],
    *,
    reason: str = "unclear",
) -> str:
    """
    Brief acknowledge + rephrase the same question.
    Used when STT is empty/weak or the answer is incomplete.
    After a few loops on the same stem, move on so candidates are not stuck.
    """
    count = int(state.get("same_question_reasks", 0) or 0) + 1
    state["same_question_reasks"] = count
    state["current_hint_level"] = evidence_ledger.bump_hint(
        int(state.get("current_hint_level", 0) or 0),
        reason="followup",
    )
    state["followup_index"] = int(state.get("followup_index", 0) or 0) + 1

    if count >= 3:
        state["same_question_reasks"] = 0
        state["current_hint_level"] = 0
        state["followup_index"] = 0
        if state.get("resume_plan"):
            state["resume_plan_index"] = int(state.get("resume_plan_index") or 0) + 1
        nxt = _next_resume_item(state)
        if nxt and str(nxt.get("question") or "").strip():
            q = llm_client._repair_resume_question(
                str(nxt.get("question") or ""),
                str(nxt.get("anchor") or ""),
            )
            return f"No problem — let's try a different angle. {q}".strip()
        return (
            "No problem — let's try a different angle. "
            "Can you describe a specific challenge you faced while working on a project "
            "and how you resolved it?"
        )

    stem = _current_qa_stem(state, hint_bump=0)
    if stem:
        stem = llm_client._repair_resume_question(stem, str((_next_resume_item(state) or {}).get("anchor") or ""))
    if reason == "empty":
        lead = "I didn't catch that clearly."
    elif reason == "weak":
        lead = "That came through incomplete."
    elif reason == "offtopic":
        lead = "I'm not sure that answers what I asked."
    else:
        lead = "Let me ask that again more simply."
    if stem:
        return f"{lead} {stem}".strip()
    return f"{lead} Could you answer the last question again in your own words?"


def _seconds_remaining(row: SessionRow) -> int:
    created = row.created_at or _now()
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    elapsed = int((_now() - created).total_seconds())
    return max(0, int(row.duration_minutes) * 60 - elapsed)


def _add_turn(db: Session, session_id: str, stage: str, role: str, content: str, meta: dict | None = None) -> TurnRow:
    seq = db.query(TurnRow).filter(TurnRow.session_id == session_id).count() + 1
    turn = TurnRow(
        session_id=session_id,
        seq=seq,
        stage=stage,
        role=role,
        content=content,
        meta_json=_dumps(meta or {}),
    )
    db.add(turn)
    db.flush()
    return turn


def _state(row: SessionRow) -> dict[str, Any]:
    return _loads(row.state_json, {})


def _save_state(row: SessionRow, state: dict[str, Any]) -> None:
    row.state_json = _dumps(state)
    row.updated_at = _now()


def _elapsed(row: SessionRow) -> int:
    return max(0, int(row.duration_minutes) * 60 - _seconds_remaining(row))


def _phase_bounds(row: SessionRow, state: dict[str, Any] | None = None) -> tuple[int, int]:
    """Return (qa_seconds, wrap_seconds).

    Default ~50% Q&A / ~5% wrap (rest coding), so a 30-minute screen gets ~15 min
    spoken technical before coding. Custom interviewers may set qa_seconds explicitly.
    """
    settings = get_settings()
    total = max(10, int(row.duration_minutes or 17)) * 60
    st = state if isinstance(state, dict) else (_loads(row.state_json, {}) or {})
    wrap_share = float(settings.wrap_share or 0.05)
    wrap = int(settings.wrap_seconds) if int(settings.wrap_seconds or 0) > 0 else max(30, int(total * wrap_share))
    qa_override = int(st.get("qa_seconds") or 0)
    if qa_override > 0:
        # Leave at least 2 minutes for coding (when enabled) + wrap.
        floor = 60
        ceiling = max(floor, total - wrap - 120)
        qa = max(floor, min(ceiling, qa_override))
    else:
        qa_share = float(settings.qa_share or 0.50)
        qa = int(settings.qa_seconds) if int(settings.qa_seconds or 0) > 0 else int(total * qa_share)
        # Floor: at least half the session (capped so coding+wrap still fit).
        half = int(total * 0.50)
        ceiling = max(60, total - wrap - 120)
        qa = max(qa, min(half, ceiling))
    return qa, wrap


def _include_coding(state: dict[str, Any]) -> bool:
    return bool(state.get("include_coding", True))


def _should_enter_coding(row: SessionRow, state: dict[str, Any]) -> bool:
    if not _include_coding(state):
        return False
    qa_secs, _wrap = _phase_bounds(row, state)
    # Conceptual-only profiles: stretch Q&A until wrap window.
    return _elapsed(row) >= qa_secs or bool(state.get("coding_after_answer"))


def _qa_agenda_complete(state: dict[str, Any]) -> bool:
    """True when configured focus topics have been visited (or there is no agenda)."""
    topics = _configured_topics(state)
    if not topics:
        return True
    covered = [str(t).strip().lower() for t in (state.get("covered_topics") or []) if str(t).strip()]
    if len(set(covered)) >= len(topics):
        return True
    cursor = int(state.get("topic_cursor", 0) or 0)
    if cursor >= max(0, len(topics) - 1) and int(state.get("topic_turns", 0) or 0) >= 1:
        return True
    return False


def _qa_budget_seconds(row: SessionRow, state: dict[str, Any]) -> int:
    """When coding is off, keep Q&A until the wrap window."""
    qa, wrap = _phase_bounds(row, state)
    if not _include_coding(state):
        total = max(10, int(row.duration_minutes or 17)) * 60
        return max(qa, total - wrap)
    return qa


def _awaiting_student_reply(db: Session, row: SessionRow) -> bool:
    last = (
        db.query(TurnRow)
        .filter(TurnRow.session_id == row.id, TurnRow.role.in_(["assistant", "student"]))
        .order_by(TurnRow.seq.desc())
        .first()
    )
    return bool(last and last.role == "assistant")


def _should_wrap(row: SessionRow, state: dict[str, Any] | None = None) -> bool:
    _qa, wrap = _phase_bounds(row, state)
    return _seconds_remaining(row) <= wrap


_LEFTOVER_QA_RE = re.compile(
    r"(?i)\b(java|dbms|sql|loan|banking|validation|blank input|operating system|"
    r"hash ?map|arraylist|normalization|acid|deadlock|semaphore|polymorphism|"
    r"tcp/ip|innodb|self[- ]intro|programming fundamentals|duplicates?|"
    r"multi[- ]?index|frequency count|primary key|foreign key|cascad|"
    r"relational|transaction|joins?|indexing|b[- ]?tree)\b"
)
_CODING_SPEECH_RE = re.compile(
    r"(?i)\b(approach|editor|complexit|edge case|nexpractice|implement|"
    r"data structure|unlock|your code|this line|tests? passed|clarifying)\b"
)


def _is_leftover_qa_caption(text: str) -> bool:
    blob = text or ""
    if _CODING_SPEECH_RE.search(blob):
        return False
    return bool(_LEFTOVER_QA_RE.search(blob))


def _ui_for(row: SessionRow, state: dict[str, Any]) -> dict[str, Any]:
    # Show NexPractice once coding round begins so they can read the prompt.
    # Keep the editor locked until the approach is accepted (stage == code/explain).
    show_editor = row.stage in {"idea", "code", "explain", "wrap"} and (
        bool(state.get("moodle_problem_id")) or bool(state.get("current_problem_id"))
    )
    editor_locked = show_editor and row.stage == "idea"
    return {
        "show_editor": show_editor,
        "editor_locked": editor_locked,
        "awaiting": state.get("awaiting", "message"),
        "can_run": show_editor and row.stage in {"code", "explain"},
        "can_submit_idea": row.stage == "idea",
        "interrupt_active": row.stage == "explain",
        "moodle_problem_id": int(state.get("moodle_problem_id") or 0),
        "problem_title": state.get("moodle_problem_title") or "",
        "problem_statement": (state.get("moodle_problem_statement") or "").strip()[:2800],
        "remount_ide": bool(state.get("_remount_ide")),
        "need_next_problem": bool(state.get("need_next_problem")),
        "used_moodle_problems": list(state.get("used_moodle_problems") or []),
        "problems_solved_count": int(state.get("problems_solved_count", 0) or 0),
        "coding_wrap_silence": bool(state.get("coding_wrap_silence")),
    }


def _current_problem(state: dict[str, Any]) -> dict[str, Any] | None:
    pid = state.get("current_problem_id")
    if not pid:
        return None
    p = get_problem(pid)
    return public_problem(p) if p else None


def _realtime_cue_for_view(row: SessionRow, state: dict[str, Any]) -> str:
    """Suppress stale Q&A coach cues once coding has started."""
    cue = str(state.get("realtime_cue") or "").strip()
    if not cue:
        return ""
    if row.stage in {"idea", "code", "explain"}:
        if _CODING_SPEECH_RE.search(cue):
            return cue[:300]
        return ""
    if row.stage == "wrap" and state.get("awaiting_coding_clarify"):
        return cue[:300]
    return cue[:300]


def session_view(db: Session, row: SessionRow) -> dict[str, Any]:
    state = _state(row)
    turns = (
        db.query(TurnRow)
        .filter(TurnRow.session_id == row.id)
        .order_by(TurnRow.seq.asc())
        .all()
    )
    report = _loads(row.report_json, None) if row.report_json else None
    return {
        "session_id": row.id,
        "status": row.status,
        "stage": row.stage,
        "duration_minutes": row.duration_minutes,
        "seconds_remaining": _seconds_remaining(row),
        "student_name": row.student_name,
        "role_track": row.role_track,
        "problem": _current_problem(state),
        "moodle_problem_id": int(state.get("moodle_problem_id") or 0),
        "ui": _ui_for(row, state),
        "scores": {
            "conceptual": state.get("score_conceptual", 0),
            "idea": state.get("score_idea", 0),
            "coding": state.get("score_coding", 0),
            "explain": state.get("score_explain", 0),
            "communication": state.get("score_communication", 0),
            "overall": row.overall_score,
        },
        "skill_graph": {k: v for k, v in (state.get("skill_graph") or {}).items() if not str(k).startswith("_")},
        "qa_topic": (
            ""
            if row.stage in {"idea", "code", "explain", "wrap"}
            else str(_focus_topic(state) or state.get("current_qa_id") or "")[:80]
        ),
        "qa_agenda": (
            []
            if row.stage in {"idea", "code", "explain", "wrap"}
            else _configured_topics(state)
        ),
        "qa_agenda_index": (
            0
            if row.stage in {"idea", "code", "explain", "wrap"}
            else int(state.get("topic_cursor", 0) or 0)
        ),
        "realtime_cue": _realtime_cue_for_view(row, state),
        "awaiting_end_confirm": bool(state.get("awaiting_end_confirm")),
        "coding_just_passed": bool(state.get("coding_just_passed")),
        "awaiting_coding_clarify": bool(state.get("awaiting_coding_clarify")),
        "explain_excerpt": str(state.get("explain_excerpt") or "")[:400],
        "voice_metrics": state.get("voice_metrics") or {},
        "turns": [
            {
                "seq": t.seq,
                "stage": t.stage,
                "role": t.role,
                "content": t.content,
                "meta": _loads(t.meta_json, {}),
            }
            for t in turns
        ],
        "report": report,
    }


def start_session(
    db: Session,
    *,
    moodle_user_id: int,
    moodle_cm_id: int,
    moodle_instance_id: int,
    student_name: str,
    role_track: str,
    duration_minutes: int,
    topics: list[str],
    resume_text: str = "",
    moodle_problem_id: int = 0,
    moodle_problem_title: str = "",
    moodle_problem_statement: str = "",
    interviewer_name: str = "NexAI",
    interviewer_style: str = "friendly",
    interviewer_briefing: str = "",
    include_coding: bool = True,
    moodle_interviewer_id: int = 0,
    difficulty: str = "intermediate",
    pace: str = "standard",
    question_mix: str = "conceptual",
    followup_depth: str = "moderate",
    avoid_topics: str = "",
    qa_minutes: int = 0,
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    resume = _normalize_resume(resume_text)
    settings = get_settings()
    duration_minutes = int(duration_minutes or settings.default_duration_minutes or 17)
    if duration_minutes < 10 or duration_minutes > 45:
        duration_minutes = 17
    # Spoken identity is always NexAI. Custom profile name is hub/display only.
    name = "NexAI"
    profile_label = (interviewer_name or "").strip()[:80]
    style = (interviewer_style or "friendly").strip().lower()
    allowed_styles = {"friendly", "strict", "brief", "socratic", "supportive", "panel"}
    if style not in allowed_styles:
        style = "friendly"
    difficulty = (difficulty or "intermediate").strip().lower()
    if difficulty not in {"beginner", "intermediate", "advanced"}:
        difficulty = "intermediate"
    pace = (pace or "standard").strip().lower()
    if pace not in {"relaxed", "standard", "brisk"}:
        pace = "standard"
    question_mix = (question_mix or "conceptual").strip().lower()
    if question_mix not in {"conceptual", "mixed", "behavioral", "system_design"}:
        question_mix = "conceptual"
    followup_depth = (followup_depth or "moderate").strip().lower()
    if followup_depth not in {"light", "moderate", "deep"}:
        followup_depth = "moderate"
    avoid = " ".join((avoid_topics or "").split())[:500]
    freeform = " ".join((interviewer_briefing or "").split())[:3500]
    # Structured custom-interviewer rules; faculty freeform briefing is highest priority.
    profile_rules = _compose_profile_rules(
        style=style,
        difficulty=difficulty,
        pace=pace,
        question_mix=question_mix,
        followup_depth=followup_depth,
        avoid_topics=avoid,
        include_coding=bool(include_coding),
        topics=topics,
    )
    if freeform and profile_rules:
        briefing = (
            "FACULTY BRIEFING (HIGHEST PRIORITY — follow these instructions closely):\n"
            + freeform
            + "\n\n"
            + profile_rules
        ).strip()[:6000]
    elif freeform:
        briefing = (
            "FACULTY BRIEFING (HIGHEST PRIORITY — follow these instructions closely):\n"
            + freeform
        ).strip()[:6000]
    elif profile_rules:
        briefing = profile_rules[:6000]
    else:
        briefing = ""
    resume_only = _is_resume_track(role_track)
    coding_on = bool(include_coding) and not resume_only
    if resume_only:
        style = "strict"
        if not freeform:
            briefing = (RESUME_BRIEFING + (("\n\n" + profile_rules) if profile_rules else "")).strip()[:6000]
        else:
            briefing = (RESUME_BRIEFING + "\n\n" + briefing).strip()[:6000]
        if not topics:
            topics = ["projects", "internships", "ownership", "impact", "stack", "tradeoffs"]
    if not coding_on:
        moodle_problem_id = 0
        moodle_problem_title = ""
        moodle_problem_statement = ""
    difficulty_ceiling = {"beginner": 1, "intermediate": 2, "advanced": 3}.get(difficulty, 2)
    max_followups = {"light": 1, "moderate": 2, "deep": 3}.get(followup_depth, 2)
    # Optional explicit Q&A minutes (custom interviewer). 0 = use default share.
    qa_mins = int(qa_minutes or 0)
    if qa_mins > 0:
        qa_mins = max(1, min(duration_minutes - 2, qa_mins))
    qa_seconds = qa_mins * 60 if qa_mins > 0 else 0
    state = {
        "topics": topics,
        "qa_index": 0,
        "qa_scores": [],
        "used_problems": [],
        "idea_attempts": 0,
        "explain_count": 0,
        "run_failures": 0,
        "flags": [],
        "awaiting": "message",
        "score_conceptual": 0,
        "score_idea": 0,
        "score_coding": 0,
        "score_explain": 0,
        "score_communication": 0,
        "current_code": "",
        "resume_text": resume,
        "resume_dossier": {},
        "resume_plan": [],
        "resume_plan_index": 0,
        "moodle_problem_id": int(moodle_problem_id or 0),
        "moodle_problem_title": (moodle_problem_title or "").strip()[:180],
        "moodle_problem_statement": (moodle_problem_statement or "").strip()[:2800],
        "skill_graph": skill_graph.default_graph(role_track),
        "claims": [],
        "voice_metrics": {},
        "voice_metric_samples": [],
        "evidence": [],
        "hint_dependency": {},
        "asked_question_ids": [],
        "current_question_id": "",
        "current_hint_level": 0,
        "difficulty_ceiling": difficulty_ceiling,
        "followup_index": 0,
        "max_followups_per_question": max_followups,
        "used_moodle_problems": [int(moodle_problem_id)] if int(moodle_problem_id or 0) else [],
        "moodle_problem_titles": [((moodle_problem_title or "").strip()[:180])] if (moodle_problem_title or "").strip() else [],
        "problems_solved_count": 0,
        "max_coding_problems": 2 if coding_on else 0,
        "include_coding": coding_on,
        "resume_only": resume_only,
        "interviewer_name": name,
        "interviewer_profile_label": profile_label,
        "interviewer_style": style,
        "interviewer_briefing": briefing,
        "moodle_interviewer_id": int(moodle_interviewer_id or 0),
        "difficulty": difficulty,
        "pace": pace,
        "question_mix": question_mix,
        "followup_depth": followup_depth,
        "avoid_topics": avoid,
        "qa_seconds": qa_seconds,
        "qa_minutes": qa_mins,
    }
    if len(resume) >= 40:
        try:
            dossier = llm_client.analyze_resume(resume)
        except Exception:
            dossier = {}
        state["resume_dossier"] = dossier or {}
        state["resume_plan"] = list((dossier or {}).get("question_plan") or [])
        state["resume_plan_index"] = 0
    row = SessionRow(
        id=session_id,
        moodle_user_id=moodle_user_id,
        moodle_cm_id=moodle_cm_id,
        moodle_instance_id=moodle_instance_id,
        student_name=student_name,
        role_track=role_track,
        status="active",
        stage="intro",
        duration_minutes=duration_minutes,
        topics_json=_dumps(topics),
        state_json=_dumps(state),
    )
    db.add(row)
    db.flush()

    # Varied intro + first conceptual question. No "say yes" gate.
    first = student_name.split()[0] if student_name else "there"
    if coding_on:
        greetings = [
            f"Hi {first}, I'm {name}. We'll do a short technical round, then one coding problem.",
            f"Hey {first}. {name} here — conceptual questions first, then you'll code.",
            f"Welcome {first}. I'm {name}, your interviewer today. Concepts first, then coding.",
            f"Good to meet you {first}. This is {name}. A few technical questions, then one problem to implement.",
            f"{first}, I'm {name}. Think of this as a live screen: questions, then one problem to implement.",
        ]
    else:
        greetings = [
            f"Hi {first}, I'm {name}. This is a resume deep-dive — no coding editor. I'll grill the work you listed.",
            f"Hey {first}. {name} here. We'll stay on your resume: projects, internships, and the stack you claim.",
            f"Welcome {first}. I'm {name}. Spoken interview only — every question comes from your resume.",
        ]
    greet = greetings[int(uuid.uuid4().int % len(greetings))]
    try:
        opening = (_begin_qa(db, row, state) or "").strip()
        if opening:
            opening = _ensure_specific_question(db, row, state, opening)
    except Exception:
        opening = ""
    if opening:
        # Prefer a single LLM (or planned) opening — never prepend a second canned intro.
        spoken = opening
        if not _opening_already_greets(spoken, name):
            spoken = f"Hi {first}, I'm {name}. {spoken}"
    else:
        # Topic-aware offline fallback (avoid random hash-map when topics/briefing exist).
        topic_hint = ""
        tops = [str(t).strip() for t in (topics or []) if str(t).strip()]
        if tops:
            topic_hint = tops[0]
        if resume_only:
            fallback_q = (
                "Walk me through the most technically demanding project on your resume — "
                "your role, the hardest bug, and how you measured success."
            )
        elif topic_hint:
            fallback_q = (
                f"Looking at {topic_hint}: what should a strong candidate explain first, "
                "and what trade-off would you call out?"
            )
        else:
            fallback_q = (
                "Tell me about a recent technical problem you solved — the constraint, "
                "your approach, and how you knew it worked."
            )
        spoken = f"{greet} {fallback_q}"
    _add_turn(db, session_id, row.stage or "qa", "assistant", spoken)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def _expire_if_needed(db: Session, row: SessionRow) -> bool:
    if row.status != "active":
        return False
    if _seconds_remaining(row) <= 0:
        finish_session(db, row, reason="time_up")
        return True
    return False


def _spoken_wrap_text(row: SessionRow, state: dict[str, Any]) -> str:
    first = (row.student_name or "there").split()[0]
    return f"Thanks for your time today, {first}. Your report is ready."


def _wants_no_more_clarify(text: str) -> bool:
    """True when the candidate declines further questions after a green coding submit."""
    t = " ".join((text or "").lower().split())
    if not t:
        return False
    if duplex_score.is_confirm_no(t):
        return True
    if duplex_score.is_confirm_yes(t) and re.search(
        r"\b(done|finish|end|wrap|close|that'?s all|all good)\b", t
    ):
        return True
    if re.search(
        r"\b(no (more )?questions?|nothing else|i'?m good|im good|all good|"
        r"that'?s (all|it|fine)|no thanks|no thank you|we can (end|wrap|finish)|"
        r"let'?s (end|wrap|finish)|i'?m done|ready to (end|finish|wrap))\b",
        t,
    ):
        return True
    if _is_end_intent(t):
        return True
    return False


def _record_explanation_from_coding_success(
    state: dict[str, Any],
    row: SessionRow,
    *,
    passed: int,
    total: int,
) -> None:
    """Credit explanation when tests pass — they demonstrated understanding in code."""
    coding = float(state.get("score_coding", 0) or 0)
    idea = float(state.get("score_idea", 0) or 0)
    ratio = passed / max(1, total)
    explain = round(min(92.0, coding * 0.56 + idea * 0.30 + ratio * 10.0 + 6.0), 1)
    if explain < 50:
        return
    state["score_explain"] = max(float(state.get("score_explain", 0) or 0), explain)
    evidence_ledger.record(
        state,
        stage="code",
        dimension="explanation",
        score=explain,
        skill="coding.explanation",
        question_id=str(state.get("moodle_problem_id") or state.get("current_problem_id") or "code"),
        hint_level=0,
        note=f"Solution validated ({passed}/{total} tests)",
        source="nexpractice",
        elapsed=_elapsed(row),
    )


def _begin_post_coding_clarify(
    db: Session,
    row: SessionRow,
    state: dict[str, Any],
    *,
    passed: int,
    total: int,
) -> dict[str, Any]:
    """After a green submit: congratulate, offer clarifications, then wrap later."""
    state["awaiting_coding_clarify"] = True
    state["coding_clarify_asks"] = 0
    state["coding_just_passed"] = True
    state["need_next_problem"] = False
    state["coding_wrap_silence"] = True
    state["realtime_cue"] = (
        f"All {passed}/{total} tests passed. Congratulate in one short sentence, then ask whether "
        "they have any clarifying questions about the problem, their solution, or the interview. "
        "Do NOT ask new coding questions, probe their code, or discuss output format. "
        "Stay silent if they have no questions. Do not invent a wrap-up yet."
    )[:300]
    row.stage = "wrap"
    state["awaiting"] = "clarify"
    spoken = (
        f"All {passed} tests passed — nice work on that. "
        "Before we close, do you have any clarifying questions about the problem or your approach?"
    )
    _add_turn(
        db,
        row.id,
        "wrap",
        "assistant",
        spoken,
        {"coding_passed": True, "coding_clarify": True},
    )
    _save_state(row, state)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def _handle_coding_clarify(
    db: Session,
    row: SessionRow,
    state: dict[str, Any],
    text: str,
    duration_sec: float,
) -> dict[str, Any]:
    """Post-pass clarification beat: answer briefly, then close when they are done."""
    _note_voice_and_claims(state, text, duration_sec)
    _add_turn(db, row.id, "wrap", "student", text, {"coding_clarify": True})
    asks = int(state.get("coding_clarify_asks", 0) or 0)
    remaining = _seconds_remaining(row)

    if _wants_no_more_clarify(text) or asks >= 2 or remaining <= 35:
        state["awaiting_coding_clarify"] = False
        state.pop("realtime_cue", None)
        _save_state(row, state)
        db.commit()
        return finish_session(db, row, reason="coding_complete")

    state["coding_clarify_asks"] = asks + 1
    if state.get("voice_duplex"):
        state["realtime_cue"] = (
            "Answer their clarification in at most two short sentences. Then ask if they have "
            "anything else before you wrap up. Do not start a new problem."
        )[:300]
        duplex_score.clear_utterance_buffer(state)
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    reply = (
        "Good question — keep that trade-off in mind. "
        "Anything else before we wrap up?"
    )
    _add_turn(db, row.id, "wrap", "assistant", reply, {"coding_clarify": True})
    _save_state(row, state)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def finish_session(db: Session, row: SessionRow, reason: str = "completed") -> dict[str, Any]:
    state = _state(row)
    if reason == "time_up" and "time_up" not in state.get("flags", []):
        state.setdefault("flags", []).append("time_up")

    turns = [
        {"stage": t.stage, "role": t.role, "content": t.content}
        for t in db.query(TurnRow).filter(TurnRow.session_id == row.id).order_by(TurnRow.seq.asc())
    ]
    student_turns = [t for t in turns if t.get("role") == "student"]
    # No padding for empty interviews — honest zeros.
    if not student_turns and not state.get("qa_scores") and int(state.get("problems_solved_count", 0) or 0) == 0:
        state["score_conceptual"] = 0.0
        state["score_idea"] = float(state.get("score_idea") or 0)
        state["score_coding"] = float(state.get("score_coding") or 0)
        state["score_explain"] = float(state.get("score_explain") or 0)
        state["score_communication"] = min(20.0, float(state.get("score_communication") or 0))
        state.setdefault("flags", []).append("no_student_answers")
    elif not state.get("qa_scores"):
        # Never invent conceptual credit when no scored Q&A happened.
        state["score_conceptual"] = float(state.get("score_conceptual") or 0)

    _resolve_pending_explain(state, row)
    # Duplex-native: rebuild floats from evidence before the report weights them.
    duplex_score.recompute_scores(state)
    duplex_score.clear_utterance_buffer(state)
    state["awaiting_end_confirm"] = False
    state["awaiting_coding_clarify"] = False

    report = build_report(state, turns)
    row.status = "completed"
    row.stage = "done"
    row.overall_score = float(report["overall_score"])
    row.recommendation = report["recommendation"]
    row.report_json = _dumps(report)
    row.ended_at = _now()
    _save_state(row, state)

    summary = _spoken_wrap_text(row, state)
    _add_turn(db, row.id, "done", "assistant", summary, {"report": True, "spoken_wrap": True})
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def log_assistant_turn(
    db: Session,
    row: SessionRow,
    content: str,
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    """
    Persist a Realtime-spoken interviewer line into the timeline.

    Duplex invents questions client-side; without this the recruiter report only
    shows student answers.
    """
    if row.status != "active":
        return session_view(db, row)
    text = " ".join((content or "").split()).strip()
    if not text or len(text) < 8:
        return session_view(db, row)
    if re.search(r"(?i)^\s*hidden\s+coach\b", text):
        return session_view(db, row)
    # Dedup against the last assistant turn.
    last = (
        db.query(TurnRow)
        .filter(TurnRow.session_id == row.id, TurnRow.role == "assistant")
        .order_by(TurnRow.seq.desc())
        .first()
    )
    if last and " ".join((last.content or "").split()).lower() == text.lower():
        return session_view(db, row)
    state = _state(row)
    # After the coding cutover, ignore leftover Q&A captions so they cannot become
    # the latest turn and hide the coding_handoff from the room.
    if row.stage in {"idea", "code", "explain"} and _is_leftover_qa_caption(text):
        return session_view(db, row)
    use_stage = (stage or row.stage or "qa").strip() or "qa"
    if use_stage == "sample":
        use_stage = "qa"
    if row.stage in {"idea", "code", "explain"} and use_stage == "qa":
        return session_view(db, row)
    if row.stage == "wrap" and state.get("awaiting_coding_clarify"):
        if _CODING_SPEECH_RE.search(text) and not re.search(
            r"(?i)\b(clarif|question|anything else|wrap|close)\b", text
        ):
            return session_view(db, row)
    _add_turn(db, row.id, use_stage, "assistant", text[:1200], {"realtime": True})
    state["voice_duplex"] = True
    # A new spoken question closes any sticky student chip buffer.
    duplex_score.clear_utterance_buffer(state)
    # Keep last spoken question for the scorer.
    if "?" in text:
        state["last_realtime_question"] = text[:280]
    _save_state(row, state)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def mark_voice_duplex(db: Session, row: SessionRow) -> None:
    """Realtime WebRTC is the mouth — engine Q&A must not persist a second interviewer."""
    state = _state(row)
    if state.get("voice_duplex"):
        return
    state["voice_duplex"] = True
    _save_state(row, state)
    db.commit()


def tick_session(db: Session, row: SessionRow) -> dict[str, Any]:
    """Apply timed phase changes. Never open coding while a technical question is unanswered."""
    if row.status != "active":
        return session_view(db, row)
    if _should_wrap(row):
        state = _state(row)
        # Let a green-submit clarification beat finish cleanly on the next message if possible.
        if state.get("awaiting_coding_clarify") and _seconds_remaining(row) > 15:
            return session_view(db, row)
        return finish_session(db, row, reason="time_up")
    state = _state(row)
    if row.stage == "qa" and _elapsed(row) >= _qa_budget_seconds(row, state):
        qa_secs = _qa_budget_seconds(row, state)
        # Always finish the live spoken question first. Realtime may still be mid-ask
        # even when the last logged turn is "student" (caption not flushed yet) — forcing
        # coding here is what makes the IDE slam in while a Q&A question is being asked.
        if not state.get("coding_after_answer"):
            state["coding_after_answer"] = True
            _save_state(row, state)
            db.commit()
            db.refresh(row)
        # Wait for the next scoreable student answer (handle_message → _close_qa_then_code).
        # Only force after a long grace if STT never arrives.
        grace = 120
        if _awaiting_student_reply(db, row):
            grace = 180
        if _elapsed(row) < qa_secs + grace:
            return session_view(db, row)
        spoken = _close_qa_then_code(db, row, state)
        state.pop("coding_handoff", None)
        _save_state(row, state)
        _add_turn(db, row.id, row.stage, "assistant", spoken, {"coding_handoff": True})
        db.commit()
        db.refresh(row)
        return session_view(db, row)
    if row.stage == "idea":
        if state.get("idea_started_elapsed") is None:
            state["idea_started_elapsed"] = _elapsed(row)
            _save_state(row, state)
            db.commit()
            db.refresh(row)
        elif _idea_seconds(row, state) >= IDEA_STAGE_MAX_SECONDS:
            # The approach probe must never hold the editor hostage, even if transcripts
            # never reached the engine.
            state["score_idea"] = max(float(state.get("score_idea", 0)), 45.0)
            spoken = _unlock_editor(
                row,
                state,
                "Let's get you coding — the editor is unlocked. Implement in NexPractice and run "
                "tests when ready. I'll ask about your logic as you go.",
            )
            _add_turn(db, row.id, row.stage, "assistant", spoken, {"coding_handoff": True})
            db.commit()
            db.refresh(row)
            return session_view(db, row)
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)
    # Snapshot the one-shot BEFORE clearing so GET/poll still sees it. Clearing first
    # made the client miss the green-submit celebration unless coding_result won a race.
    view = session_view(db, row)
    state = _state(row)
    if state.get("coding_just_passed"):
        state["coding_just_passed"] = False
        _save_state(row, state)
        db.commit()
    return view


def _recent_transcript(db: Session, session_id: str, limit: int = 8) -> list[dict[str, str]]:
    turns = (
        db.query(TurnRow)
        .filter(TurnRow.session_id == session_id, TurnRow.role.in_(["assistant", "student"]))
        .order_by(TurnRow.seq.desc())
        .limit(limit)
        .all()
    )
    turns = list(reversed(turns))
    return [{"role": t.role, "content": t.content} for t in turns]


def _apply_score_communication(state: dict[str, Any], answer: str, score: float) -> None:
    # Admitting "I don't know" earns no communication credit.
    if llm_client.is_no_knowledge_answer(answer):
        prev = float(state.get("score_communication") or 0)
        state["score_communication"] = 0.0 if prev <= 0 else round(prev * 0.5, 1)
        return
    words = len(answer.split())
    state["score_communication"] = min(
        100.0,
        45 + min(35, words / 2) + (10 if score >= 60 else 0),
    )


def _extract_claims(answer: str) -> list[str]:
    text = " ".join((answer or "").split())
    claims: list[str] = []
    patterns = [
        r"\bI have (?:worked|used|built|designed|implemented|experience)[^.!?]{0,80}",
        r"\bI(?:'ve| have) (?:\d+\+?\s*years?)[^.!?]{0,60}",
        r"\bwe used [^.!?]{0,60}",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            claim = m.group(0).strip()
            if len(claim) >= 12:
                claims.append(claim[:160])
    return claims[:3]


def _note_voice_and_claims(state: dict[str, Any], answer: str, duration_sec: float = 0.0) -> None:
    metrics = voice_metrics.analyze_utterance(answer, duration_sec=duration_sec or None)
    samples = state.setdefault("voice_metric_samples", [])
    samples.append(metrics)
    state["voice_metric_samples"] = samples[-30:]
    # Aggregate latest rolling snapshot for the report.
    state["voice_metrics"] = {
        "speaking_rate_wpm": metrics["speaking_rate_wpm"],
        "filler_count_total": sum(int(s.get("filler_count", 0)) for s in state["voice_metric_samples"]),
        "clarity": round(
            sum(float(s.get("clarity", 0)) for s in state["voice_metric_samples"]) / max(1, len(state["voice_metric_samples"])),
            1,
        ),
        "structure": round(
            sum(float(s.get("structure", 0)) for s in state["voice_metric_samples"]) / max(1, len(state["voice_metric_samples"])),
            1,
        ),
        "samples": len(state["voice_metric_samples"]),
        "latest": metrics,
    }
    if llm_client.is_no_knowledge_answer(answer):
        # Don't let voice metrics inflate communication for "I don't know".
        prev = float(state.get("score_communication") or 0)
        state["score_communication"] = 0.0 if prev <= 0 else round(min(prev, 10.0) * 0.5, 1)
    else:
        state["score_communication"] = voice_metrics.blend_communication(
            float(state.get("score_communication") or 0),
            metrics,
        )
    for claim in _extract_claims(answer):
        bag = state.setdefault("claims", [])
        if claim not in bag:
            bag.append(claim)
        state["claims"] = bag[-20:]


def _configured_topics(state: dict[str, Any]) -> list[str]:
    return [str(t).strip() for t in (state.get("topics") or []) if str(t).strip()][:12]


def _resume_questions_allowed(row: SessionRow, state: dict[str, Any]) -> bool:
    """Resume-anchored questions when the candidate supplied a CV or faculty asked for them."""
    if _is_resume_track(row.role_track, state):
        return True
    # Custom topic lists own the run-of-show. Only grill the CV during a
    # self-introduction / resume agenda item — otherwise Realtime hijacks DBMS/DSA.
    if _topic_locked(row, state):
        focus = _focus_topic(state).lower()
        return any(k in focus for k in ("intro", "self", "resume", "cv"))
    resume = (state.get("resume_text") or "").strip()
    if len(resume) >= 80 or state.get("resume_plan"):
        return True
    briefing = (state.get("interviewer_briefing") or "").lower()
    return any(k in briefing for k in ("resume", "cv ", " cv", "their projects", "past projects"))


def _resume_plan_quota(state: dict[str, Any]) -> int:
    """How many resume-plan items to weave into a mixed (non-resume-only) interview."""
    plan = list(state.get("resume_plan") or [])
    if not plan:
        return 0
    return min(max(3, len(plan)), 6)


def _take_planned_resume_question(state: dict[str, Any]) -> str | None:
    """
    Pull the next resume-plan question when the candidate uploaded a CV.

    Used on technical tracks so projects/internships are grilled, not only generic topics.
    """
    plan = list(state.get("resume_plan") or [])
    if not plan:
        return None
    idx = int(state.get("resume_plan_index") or 0)
    if idx >= len(plan) or idx >= _resume_plan_quota(state):
        return None
    item = plan[idx]
    state["resume_plan_index"] = idx + 1
    q = str(item.get("question") or "").strip()
    if not q:
        return None
    anchor = str(item.get("anchor") or "")
    return llm_client._repair_resume_question(q, anchor)


def _topic_locked(row: SessionRow, state: dict[str, Any]) -> bool:
    """True when a custom / briefed interview must stay inside its configured topics."""
    if _is_resume_track(row.role_track, state):
        return False
    dynamic_ok = (
        bool(state.get("interviewer_briefing"))
        or bool(state.get("moodle_interviewer_id"))
    )
    return bool(_configured_topics(state)) and dynamic_ok


def _focus_topic(state: dict[str, Any]) -> str:
    """Current topic from the faculty list, rotated by the topic cursor."""
    topics = _configured_topics(state)
    if not topics:
        return ""
    cursor = int(state.get("topic_cursor", 0) or 0)
    return topics[cursor % len(topics)]


# Consecutive questions allowed on one configured topic before rotating away.
MAX_TURNS_PER_TOPIC = 3


def _advance_focus_topic(state: dict[str, Any]) -> str:
    """Move to the next configured topic; returns the new focus topic."""
    topics = _configured_topics(state)
    if not topics:
        state["topic_turns"] = 0
        return ""
    current = _focus_topic(state)
    covered = state.setdefault("covered_topics", [])
    if current and current not in covered:
        covered.append(current)
    state["topic_cursor"] = int(state.get("topic_cursor", 0) or 0) + 1
    state["topic_turns"] = 0
    return _focus_topic(state)


def _note_topic_turn(state: dict[str, Any]) -> None:
    """Count questions asked on the current topic and rotate when it is exhausted."""
    if not _configured_topics(state):
        return
    turns = int(state.get("topic_turns", 0) or 0) + 1
    state["topic_turns"] = turns
    if turns >= MAX_TURNS_PER_TOPIC:
        _advance_focus_topic(state)


def _move_on_bridge(state: dict[str, Any]) -> str:
    """
    Short, non-judgemental bridge used when the engine changes topic.

    Never mentions scoring, weakness, or readiness — that is what made earlier
    sessions read as rude.
    """
    style = str(state.get("interviewer_style") or "friendly").strip().lower()
    variants = {
        "friendly": [
            "No problem — let's try a different angle.",
            "All good, let's switch things up.",
            "That's fine — here's something else.",
        ],
        "supportive": [
            "That's completely fine — let's move on.",
            "No worries at all, next one.",
        ],
        "strict": ["Let's move on.", "Next area."],
        "brief": ["Moving on.", "Next one."],
        "socratic": ["Let's come at this from another side.", "Different angle then."],
        "panel": ["Understood — let's change topic.", "Fine, next area."],
    }.get(style, ["Let's move on to something else.", "Thanks — next area."])
    idx = int(state.get("bridge_index", 0) or 0)
    state["bridge_index"] = idx + 1
    return variants[idx % len(variants)]


_GENERIC_SKILL_TAGS = {
    "self-introduction",
    "self introduction",
    "opening",
    "intro",
    "introduction",
}


def _evidence_skill(state: dict[str, Any], node: dict[str, Any] | None, extra: str = "") -> str:
    """Skill tag for the evidence ledger — never leave it stuck on 'self-introduction'."""
    if node:
        try:
            return f"{node['skill'][0]}.{node['skill'][1]}"
        except Exception:
            pass
    tag = str(extra or state.get("current_qa_id") or _focus_topic(state) or "").strip()
    if tag.lower() in _GENERIC_SKILL_TAGS:
        tag = _focus_topic(state) or tag
    return tag[:80]


def _ensure_specific_question(
    db: Session,
    row: SessionRow,
    state: dict[str, Any],
    reply: str,
) -> str:
    """
    Last-line guard: never speak a rude, scored, vague, or off-lock DSA question.

    Does not advance the topic cursor — the engine already chose the topic.
    """
    # Coding / wrap lines mention data structures on purpose. Replacing them with
    # another Java/DBMS stem is how the voice kept Q&A going after the IDE opened.
    if row.stage in {"idea", "code", "explain", "wrap"} or state.get("coding_handoff"):
        return llm_client.strip_spoken_meta(reply or "") or (reply or "")
    topics = _configured_topics(state)
    text = llm_client.strip_spoken_meta(reply or "")
    if (
        text
        and not llm_client.is_vague_question(text)
        and not llm_client.is_off_lock_dsa(text, topics)
    ):
        return text
    generated = None
    try:
        generated = llm_client.topic_question(
            role_track=row.role_track,
            topics=topics or (state.get("topics") or []),
            focus_topic=_focus_topic(state),
            difficulty=str(state.get("difficulty") or "intermediate"),
            interviewer_style=str(state.get("interviewer_style") or "friendly"),
            interviewer_briefing=str(state.get("interviewer_briefing") or ""),
            asked_questions=_asked_question_texts(db, row),
            transcript=_recent_transcript(db, row.id, limit=6),
            bridge_hint=_move_on_bridge(state),
            resume_questions_allowed=_resume_questions_allowed(row, state),
        )
    except Exception:
        generated = None
    if generated and generated.get("reply"):
        cand = llm_client.strip_spoken_meta(str(generated["reply"]))
        if (
            cand
            and not llm_client.is_vague_question(cand)
            and not llm_client.is_off_lock_dsa(cand, topics)
        ):
            if generated.get("topic_tag"):
                state["current_qa_id"] = str(generated["topic_tag"])[:80]
            elif _focus_topic(state):
                state["current_qa_id"] = _focus_topic(state)[:80]
            _save_state(row, state)
            return cand
    if _focus_topic(state):
        state["current_qa_id"] = _focus_topic(state)[:80]
    stem = _offline_topic_question(state, None)
    lead = _move_on_bridge(state)
    _save_state(row, state)
    return f"{lead} {stem}".strip()


def _asked_question_texts(db: Session, row: SessionRow, limit: int = 8) -> list[str]:
    """Recent interviewer questions, so new questions do not repeat them."""
    out: list[str] = []
    try:
        for text in _last_assistant_texts(db, row.id, limit=limit):
            clean = " ".join(str(text or "").split())
            if "?" in clean:
                out.append(clean[:220])
    except Exception:
        return []
    return list(reversed(out))


def _llm_context(row: SessionRow, state: dict[str, Any], **extra: Any) -> dict[str, Any]:
    graph = state.get("skill_graph") or skill_graph.default_graph(row.role_track)
    qid = state.get("current_question_id") or ""
    node = question_graph.get_node(qid)
    hint = int(state.get("current_hint_level", 0) or 0)
    briefing = state.get("interviewer_briefing") or ""
    resume_only = _is_resume_track(row.role_track, state)
    dynamic_ok = bool(briefing) or bool(state.get("moodle_interviewer_id")) or resume_only
    suggested = "concept" if resume_only else question_graph.suggested_format_for_turn(
        int(state.get("qa_index", 0) or 0),
        briefing,
    )
    resume_ok = _resume_questions_allowed(row, state)
    topics_cfg = _configured_topics(state)
    # Custom / briefed interviews: hide the DSA skill graph so the model cannot drift
    # into hash-map / complexity questions that were never configured.
    graph_summary = None if (dynamic_ok and not resume_only and topics_cfg) else skill_graph.summarize_for_llm(graph)
    ctx = {
        "seconds_remaining": _seconds_remaining(row),
        "qa_index": state.get("qa_index", 0),
        "asked_topics": state.get("asked_topics", []),
        "resume_text": (state.get("resume_text") or "") if resume_ok else "",
        "resume_questions_allowed": resume_ok,
        "allowed_topics": topics_cfg,
        "focus_topic": _focus_topic(state),
        "moodle_problem_id": state.get("moodle_problem_id"),
        "skill_graph_summary": graph_summary,
        "claims": state.get("claims") or [],
        "voice_metrics_latest": (state.get("voice_metrics") or {}).get("latest"),
        "current_hint_level": hint,
        "difficulty_ceiling": int(state.get("difficulty_ceiling", 2) or 2),
        "question_node": question_graph.node_context_for_llm(
            node, hint_level=hint, dynamic_ok=dynamic_ok
        ),
        "interviewer_name": state.get("interviewer_name") or "NexAI",
        "interviewer_style": state.get("interviewer_style") or "friendly",
        "interviewer_briefing": briefing,
        "include_coding": _include_coding(state),
        "dynamic_question_ok": dynamic_ok,
        "suggested_format": suggested,
        "resume_only": resume_only,
        "resume_dossier": (state.get("resume_dossier") or {}) if resume_ok else {},
        "must_ask_next": _next_resume_item(state) if resume_ok else None,
        "difficulty": state.get("difficulty") or "intermediate",
        "pace": state.get("pace") or "standard",
        "question_mix": state.get("question_mix") or "conceptual",
        "followup_depth": state.get("followup_depth") or "moderate",
        "avoid_topics": state.get("avoid_topics") or "",
    }
    ctx.update(extra)
    return ctx


def _activate_question(state: dict[str, Any], node: dict[str, Any] | None) -> None:
    if not node:
        return
    state["current_question_id"] = node["id"]
    state["current_hint_level"] = 0
    state["followup_index"] = 0
    state["same_question_reasks"] = 0
    state["current_qa_id"] = f"{node['skill'][0]}.{node['skill'][1]}"
    ids = state.setdefault("asked_question_ids", [])
    if node["id"] not in ids:
        ids.append(node["id"])
    topics = state.setdefault("asked_topics", [])
    tag = f"{node['skill'][0]}.{node['skill'][1]}"
    if tag not in topics:
        topics.append(tag)


# Concrete offline stems per topic keyword. Used only when the LLM cannot answer —
# they still have to sound like a real interviewer question, not a template.
_TOPIC_STEMS: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("java", "jvm", "spring"),
        [
            "In Java, you loop over an ArrayList of orders and call remove() on the cancelled ones "
            "inside that loop. What happens at runtime, and what would you do instead?",
            "You have a Java class used as a HashMap key and you override equals but not hashCode. "
            "What breaks when you look values up, and why?",
        ],
    ),
    (
        ("python",),
        [
            "In Python, a function takes an empty list as a default argument and appends to it. "
            "Call it three times — what do you see, and why?",
            "You need to strip duplicates from a 2 million row CSV in Python without blowing memory. "
            "What do you reach for, and where does it get expensive?",
        ],
    ),
    (
        ("oop", "object oriented", "object-oriented", "design principle", "solid"),
        [
            "You have a PaymentProcessor class that handles cards, wallets, and refunds in one file. "
            "How would you break it up, and what specifically improves?",
            "Give me one case from your own code where inheritance was the wrong call and composition "
            "would have been better. What went wrong?",
        ],
    ),
    (
        ("database", "sql", "postgres", "mysql", "query"),
        [
            "A report query on a 5 million row orders table takes 40 seconds. Walk me through the first "
            "two things you check, and what you would change.",
            "You add an index and the query gets slower. Give me one concrete reason that happens.",
        ],
    ),
    (
        ("data structure", "algorithm", "dsa", "stack", "queue", "tree", "graph"),
        [
            "Your array-backed queue has 100 slots. After 100 enqueues and 60 dequeues an enqueue fails "
            "even though 60 slots are free. What is going on, and how do you fix it?",
            "You are matching brackets in a 50 000 character file. Which structure do you use, and what "
            "input makes a naive version fail?",
        ],
    ),
    (
        ("web", "react", "frontend", "javascript", "api", "rest", "node"),
        [
            "A user double-clicks Submit and the order is created twice. Where do you fix that, and why "
            "there rather than the button?",
            "Your list page refetches everything after each edit and feels sluggish. What would you change "
            "first, and what is the trade-off?",
        ],
    ),
    (
        ("system design", "architecture", "scalab", "microservice", "cloud", "devops"),
        [
            "One service writes to the database and another reads it two seconds later and sees stale data. "
            "How would you handle that, and what do you give up?",
            "Your service is fine at 100 requests a second and falls over at 500. What do you measure first, "
            "and what is your first change?",
        ],
    ),
    (
        ("testing", "quality", "debug"),
        [
            "A test passes locally and fails in CI about one run in five. How do you track that down?",
            "You have 20 minutes to test a new discount-code feature. What do you actually test, and what "
            "do you deliberately skip?",
        ],
    ),
]


def _offline_topic_question(state: dict[str, Any], node: dict[str, Any] | None) -> str:
    """Fallback question when the LLM is unavailable — stay on the configured topics."""
    focus = _focus_topic(state)
    variant = int(state.get("offline_q_index", 0) or 0)
    state["offline_q_index"] = variant + 1
    if focus:
        low = focus.lower()
        for keys, stems in _TOPIC_STEMS:
            if any(k in low for k in keys):
                return stems[variant % len(stems)]
        return (
            f"Staying on {focus}: describe one specific situation where you used it, the decision you "
            "had to make, and what nearly went wrong."
        )
    if node and node.get("stem"):
        return str(node["stem"])
    return (
        "Tell me about one technical problem you fixed recently — what was actually broken, "
        "and how did you find it?"
    )


def _next_topic_turn(
    db: Session,
    row: SessionRow,
    state: dict[str, Any],
    *,
    bridge: str = "",
) -> str:
    """
    Change topic with a real, specific question.

    The engine decides to move on in several places (weak answer, exhausted
    follow-up budget, declined question). Every one of those used to speak a
    canned "let's move to the next topic" line, so this is the single path that
    turns that decision into an actual interview question.
    """
    lead = (bridge or _move_on_bridge(state)).strip()
    focus = _advance_focus_topic(state)
    state["current_question_id"] = ""
    state["followup_index"] = 0
    state["current_hint_level"] = 0
    state["respecify_count"] = 0
    _note_topic_turn(state)
    if focus:
        state["current_qa_id"] = focus[:80]
    if state.get("voice_duplex"):
        state["realtime_cue"] = (
            f"{lead} Ask ONE concrete question on AGENDA item '{focus or 'the next listed topic'}'. "
            "Stay inside this topic. Never invent a product they did not mention. "
            "One question, then wait."
        )[:300]
        _save_state(row, state)
        return ""
    generated = None
    try:
        generated = llm_client.topic_question(
            role_track=row.role_track,
            topics=_configured_topics(state) or (state.get("topics") or []),
            focus_topic=focus,
            difficulty=str(state.get("difficulty") or "intermediate"),
            interviewer_style=str(state.get("interviewer_style") or "friendly"),
            interviewer_briefing=str(state.get("interviewer_briefing") or ""),
            asked_questions=_asked_question_texts(db, row),
            transcript=_recent_transcript(db, row.id, limit=6),
            bridge_hint=lead,
            resume_questions_allowed=_resume_questions_allowed(row, state),
        )
    except Exception:
        generated = None

    resume_only = _is_resume_track(row.role_track, state)
    if generated and generated.get("reply"):
        if generated.get("topic_tag"):
            state["current_qa_id"] = str(generated["topic_tag"])[:80]
        elif focus:
            state["current_qa_id"] = focus[:80]
        _save_state(row, state)
        return str(generated["reply"]).strip()

    if not resume_only and _resume_questions_allowed(row, state):
        planned = _take_planned_resume_question(state)
        if planned:
            _save_state(row, state)
            if lead and not lead.endswith((".", "!", "?")):
                lead += "."
            return f"{lead} {planned}".strip() if lead else planned

    if focus:
        state["current_qa_id"] = focus[:80]
    question = _offline_topic_question(state, None)
    _save_state(row, state)
    if lead and not lead.endswith((".", "!", "?")):
        lead += "."
    return f"{lead} {question}".strip()


def _begin_qa(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    row.stage = "qa"
    state["qa_index"] = 0
    state["awaiting"] = "message"
    state.setdefault("asked_topics", [])
    state.setdefault("asked_question_ids", [])
    state.setdefault("difficulty_ceiling", 2)

    topics = state.get("topics") or _loads(row.topics_json, [])
    briefing = state.get("interviewer_briefing") or ""
    resume_only = _is_resume_track(row.role_track, state)
    custom_id = int(state.get("moodle_interviewer_id") or 0)
    dynamic_ok = bool(briefing) or bool(custom_id) or resume_only
    suggested = "concept" if resume_only else question_graph.suggested_format_for_turn(0, briefing)
    # Custom interviewers: skip bank opener so hashmap stems cannot bias the first question.
    node = None
    if not resume_only and custom_id <= 0:
        node = question_graph.pick_opening(
            row.role_track,
            topics,
            briefing=briefing,
            prefer_format=None,
        )
    if node:
        _activate_question(state, node)
    node_ctx = question_graph.node_context_for_llm(node, hint_level=0, dynamic_ok=dynamic_ok) if node else None

    # Resume track: do not wait on a second LLM call during /sessions/start
    # (analyze + opening was exceeding Railway/Moodle timeouts → "Interview service error").
    if resume_only:
        nxt = _next_resume_item(state)
        planned = str((nxt or {}).get("question") or "").strip()
        if planned:
            first = (row.student_name or "there").split()[0]
            name = str(state.get("interviewer_name") or "NexAI")
            state["llm_mode"] = True
            state["resume_plan_index"] = int(state.get("resume_plan_index") or 0) + 1
            _save_state(row, state)
            return f"Hi {first}, I'm {name}. I read your resume. {planned}"

    resume_ok = _resume_questions_allowed(row, state)
    dynamic = llm_client.first_question(
        role_track=row.role_track,
        topics=topics,
        resume_text=(state.get("resume_text") or "") if resume_ok else "",
        question_node=node_ctx,
        interviewer_name=str(state.get("interviewer_name") or "NexAI"),
        interviewer_style=str(state.get("interviewer_style") or "friendly"),
        interviewer_briefing=briefing,
        include_coding=_include_coding(state),
        suggested_format=suggested,
        resume_only=resume_only,
        resume_dossier=(state.get("resume_dossier") or {}) if resume_ok else {},
        must_ask_next=_next_resume_item(state) if resume_ok else None,
        dynamic_question_ok=dynamic_ok,
        difficulty=str(state.get("difficulty") or "intermediate"),
        pace=str(state.get("pace") or "standard"),
        question_mix=str(state.get("question_mix") or "conceptual"),
        followup_depth=str(state.get("followup_depth") or "moderate"),
        avoid_topics=str(state.get("avoid_topics") or ""),
        focus_topic=_focus_topic(state),
        resume_questions_allowed=resume_ok,
    )
    if dynamic:
        if dynamic.get("question_id"):
            state["current_question_id"] = dynamic["question_id"]
        elif dynamic_ok:
            # LLM invented the opener — keep skill tag soft, clear rigid bank lock.
            state["current_question_id"] = ""
            if dynamic.get("topic_tag"):
                tag = str(dynamic["topic_tag"])[:80]
                if tag.lower() in _GENERIC_SKILL_TAGS:
                    tag = _focus_topic(state) or tag
                state["current_qa_id"] = tag
            elif _focus_topic(state):
                state["current_qa_id"] = _focus_topic(state)[:80]
        state["llm_mode"] = True
        if resume_only:
            state["resume_plan_index"] = int(state.get("resume_plan_index") or 0) + 1
        _note_topic_turn(state)
        _save_state(row, state)
        return dynamic["reply"]

    if llm_client.llm_configured():
        err = llm_client.last_error() or "unknown LLM error"
        try:
            logger = __import__("logging").getLogger("interview.orch")
            logger.warning("Opening LLM failed: %s", err[:300])
        except Exception:
            pass
        # Never speak Railway/JSON internals to the candidate.
        state["llm_mode"] = False
        _save_state(row, state)
        if resume_only:
            nxt = _next_resume_item(state)
            if nxt and nxt.get("question"):
                return str(nxt["question"])
            return (
                "Thanks for the resume. Walk me through the most technically demanding project on it — "
                "your role, the hardest bug, and how you measured success."
            )
        return f"Let's begin. {_offline_topic_question(state, node)}"

    # Offline/dev fallback: speak the curated stem directly.
    state["llm_mode"] = False
    _save_state(row, state)
    if resume_only:
        nxt = _next_resume_item(state)
        if nxt and nxt.get("question"):
            return str(nxt["question"])
        return (
            "Walk me through the most technically demanding project on your resume — "
            "your role, the hardest bug, and how you measured success."
        )
    return f"First question: {_offline_topic_question(state, node)} Give a clear structured answer."


def _next_qa_or_coding(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    topics = state.get("topics") or _loads(row.topics_json, [])
    idx = int(state.get("qa_index", 0))
    graph = state.get("skill_graph") or skill_graph.default_graph(row.role_track)
    qid = state.get("current_question_id") or ""
    node = question_graph.get_node(qid)
    skill_tag = _evidence_skill(state, node)
    hint_now = int(state.get("current_hint_level", 0) or 0)

    if llm_client.is_skip_topic_request(answer):
        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=0.0,
            skill=skill_tag,
            question_id=qid,
            hint_level=hint_now,
            note="Candidate asked to skip this topic",
            source="gate",
            elapsed=_elapsed(row),
        )
        return _next_topic_turn(
            db,
            row,
            state,
            bridge="Sure — next area.",
        )

    # The candidate is objecting to the QUESTION, not declining the topic.
    # Runs before the no-knowledge gate ("I don't know what you mean by that question"
    # used to score 0 and trigger a canned move-on).
    if llm_client.is_unclear_question_response(answer):
        repairs = int(state.get("respecify_count", 0) or 0)
        state["respecify_count"] = repairs + 1
        last_q = ""
        try:
            prev = _last_assistant_texts(db, row.id, limit=1)
            last_q = prev[0] if prev else ""
        except Exception:
            last_q = ""
        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=0.0,
            skill=skill_tag,
            question_id=qid,
            hint_level=int(state.get("current_hint_level", 0) or 0),
            note="Candidate found the question unclear — re-specified",
            source="gate",
            elapsed=_elapsed(row),
        )
        # One failed rephrase is enough — looping an invented scenario (loans, etc.) is worse.
        if repairs >= 1:
            return _next_topic_turn(
                db,
                row,
                state,
                bridge="Fair enough — let me ask something clearer.",
            )
        if state.get("voice_duplex"):
            state["realtime_cue"] = (
                "They did not understand. Rephrase ONCE using nouns THEY said. "
                "Drop any invented scenario. Then wait. One question only."
            )[:300]
            _save_state(row, state)
            return ""
        fixed = None
        try:
            fixed = llm_client.respecify_question(
                last_question=last_q,
                candidate_reply=answer,
                topics=_configured_topics(state) or topics,
                focus_topic=_focus_topic(state),
                difficulty=str(state.get("difficulty") or "intermediate"),
                interviewer_style=str(state.get("interviewer_style") or "friendly"),
                interviewer_briefing=str(state.get("interviewer_briefing") or ""),
                asked_questions=_asked_question_texts(db, row),
            )
        except Exception:
            fixed = None
        if fixed:
            state["current_hint_level"] = evidence_ledger.bump_hint(hint_now, reason="followup")
            state["followup_index"] = int(state.get("followup_index", 0) or 0) + 1
            _save_state(row, state)
            return fixed
        return _next_topic_turn(
            db,
            row,
            state,
            bridge="That's on me — let me make it concrete.",
        )

    # Explicit "I don't know" / not aware — score exactly 0, mark skill weak, move on.
    # Must run before the weak-answer gate (short "idk" would otherwise get soft credit).
    if llm_client.is_no_knowledge_answer(answer):
        score = 0.0
        state.setdefault("qa_scores", []).append(score)
        state.setdefault("no_knowledge_count", 0)
        state["no_knowledge_count"] = int(state["no_knowledge_count"]) + 1
        state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
        _apply_score_communication(state, answer, score)
        state["qa_index"] = idx + 1
        state["current_hint_level"] = max(
            2, evidence_ledger.bump_hint(hint_now, reason="followup")
        )
        state["skill_graph"] = skill_graph.update_skill(
            graph,
            topic_tag=skill_tag or state.get("current_qa_id") or "",
            score_0_100=score,
            evidence=f"No-knowledge: {(answer or '')[:120]}",
            weight=0.85,
        )
        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=score,
            skill=skill_tag,
            question_id=qid,
            hint_level=int(state["current_hint_level"]),
            note=(answer or "")[:160],
            source="no_knowledge",
            elapsed=_elapsed(row),
        )
        if _should_enter_coding(row, state) or _elapsed(row) >= _qa_budget_seconds(row, state):
            state["coding_after_answer"] = False
            _save_state(row, state)
            return _close_qa_then_code(db, row, state)

        # Topic-locked custom interviews never fall back to the DSA bank — that is
        # how sessions drifted into Big-O / hash-map stems that were never configured.
        nxt = None if _topic_locked(row, state) else question_graph.pick_next(
            role_track=row.role_track,
            graph=state["skill_graph"],
            asked_ids=list(state.get("asked_question_ids") or []),
            difficulty_ceiling=max(1, int(state.get("difficulty_ceiling", 2) or 2) - 1),
            topics=topics,
            briefing=state.get("interviewer_briefing") or "",
        )
        if nxt and not state.get("voice_duplex"):
            _activate_question(state, nxt)
            spoken = question_graph.spoken_prompt(nxt, hint_level=0)
            _save_state(row, state)
            return f"{_move_on_bridge(state)} {spoken}".strip()
        return _next_topic_turn(db, row, state)

    if llm_client.is_weak_answer(answer):
        # Soft H1 clarify without advancing the topic — rephrase the same question.
        state["current_hint_level"] = evidence_ledger.bump_hint(hint_now, reason="followup")
        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=0.0,
            skill=skill_tag,
            question_id=qid,
            hint_level=state["current_hint_level"],
            note="Weak/filler utterance — clarification requested",
            source="gate",
            elapsed=_elapsed(row),
        )
        _save_state(row, state)
        if state.get("voice_duplex"):
            state["realtime_cue"] = (
                "Their last answer was thin. Re-ask the SAME AGENDA item once, using their nouns. "
                "One question, then wait."
            )[:300]
            _save_state(row, state)
            return ""
        return _reask_current_question(state, reason="weak")

    # Half-cut / trailing STT — rephrase instead of scoring a partial thought.
    if llm_client.looks_incomplete_answer(answer):
        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=10.0,
            skill=skill_tag,
            question_id=qid,
            hint_level=int(state.get("current_hint_level", 0) or 0),
            note="Incomplete utterance — re-ask",
            source="gate",
            elapsed=_elapsed(row),
        )
        reply = _reask_current_question(state, reason="unclear")
        if state.get("voice_duplex"):
            state["realtime_cue"] = (
                "That answer was cut off. Re-ask the SAME AGENDA item once. One question, then wait."
            )[:300]
            reply = ""
        _save_state(row, state)
        return reply

    # Time to close Q&A: do not ask another conceptual question, and skip the LLM for speed.
    if _elapsed(row) >= _qa_budget_seconds(row, state) or state.get("coding_after_answer"):
        state["qa_index"] = idx + 1
        state["coding_after_answer"] = False
        _apply_score_communication(state, answer, 60)
        _save_state(row, state)
        return _close_qa_then_code(db, row, state)

    # Hidden technical observer → steers adaptive probes (Chakra-style).
    last_q = ""
    try:
        prev = _last_assistant_texts(db, row.id, limit=1)
        last_q = prev[0] if prev else ""
    except Exception:
        last_q = ""
    if not last_q:
        last_q = str(state.get("last_realtime_question") or "")

    llm_result = llm_client.score_turn(
        stage="qa",
        role_track=row.role_track,
        topics=topics,
        last_question=last_q,
        student_message=answer,
        asked_questions=_asked_question_texts(db, row),
        briefing=str(state.get("interviewer_briefing") or ""),
        current_topic=str(state.get("current_qa_id") or _focus_topic(state) or ""),
        include_coding=_include_coding(state),
        probes_on_topic=int(state.get("followup_index", 0) or 0),
    )

    if llm_result:
        # Thin answers should not look stronger than they are.
        model_score = float(llm_result["score"])
        if llm_result.get("depth") == "superficial":
            model_score = min(model_score, 45.0)
        score = llm_client.clamp_answer_score(answer, model_score)
        action = llm_result["next_action"]
        if llm_result.get("depth") in {"superficial", "partial"} and action == "next_topic":
            max_fu = int(state.get("max_followups_per_question", 2) or 2)
            if int(state.get("followup_index", 0) or 0) < max_fu:
                action = "followup"
                llm_result["next_action"] = "followup"
        reported_hint = int(llm_result.get("hint_level", hint_now) or hint_now)

        # Off-topic / very weak: never advance — force a rephrase follow-up
        # (unless follow-up depth budget is exhausted → move on).
        max_fu = int(state.get("max_followups_per_question", 2) or 2)
        fu_count = int(state.get("followup_index", 0) or 0)
        if action != "followup" and (
            score < 40
            or llm_client.looks_incomplete_answer(answer)
            or llm_client.is_weak_answer(answer)
        ):
            if fu_count < max_fu:
                action = "followup"
            else:
                action = "next_topic"
                llm_result["next_action"] = "next_topic"
                state["force_fresh_question"] = True

        # Never advance on weak LLM scores that re-ask.
        if action == "followup" and score < 55:
            if fu_count >= max_fu:
                # Depth budget used — accept score and move topic instead of looping.
                action = "next_topic"
                llm_result["next_action"] = "next_topic"
                state["force_fresh_question"] = True
            else:
                new_hint = evidence_ledger.bump_hint(
                    max(hint_now, reported_hint),
                    reason="followup",
                )
                state["current_hint_level"] = new_hint
                state["followup_index"] = fu_count + 1
                state["skill_graph"] = skill_graph.update_skill(
                    graph,
                    topic_tag=skill_tag or llm_result.get("topic_tag") or "",
                    score_0_100=score,
                    evidence=f"Follow-up H{new_hint}: {(answer or '')[:120]}",
                    weight=0.55 if score < 25 else 0.35,
                )
                evidence_ledger.record(
                    state,
                    stage="qa",
                    dimension="conceptual",
                    score=score,
                    skill=skill_tag,
                    question_id=qid,
                    hint_level=new_hint,
                    note=(answer or "")[:160],
                    source="llm",
                    elapsed=_elapsed(row),
                )
                if llm_result.get("topic_tag"):
                    state["current_qa_id"] = str(llm_result["topic_tag"])[:80]
                # Realtime invents the spoken follow-up — never store coach cues or bank stems.
                _save_state(row, state)
                return ""

        if action == "next_topic":
            state["followup_index"] = 0
            state["current_hint_level"] = 0

        state.setdefault("qa_scores", []).append(score)
        state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
        _apply_score_communication(state, answer, score)
        state["qa_index"] = idx + 1
        if _resume_questions_allowed(row, state) and action != "followup":
            # Resume-only tracks advance every turn; mixed tracks use _take_planned_resume_question.
            if _is_resume_track(row.role_track, state):
                state["resume_plan_index"] = int(state.get("resume_plan_index") or 0) + 1
        state["llm_mode"] = True
        state["difficulty_ceiling"] = question_graph.adjust_difficulty_ceiling(
            int(state.get("difficulty_ceiling", 2) or 2),
            score,
        )

        evidence_ledger.record(
            state,
            stage="qa",
            dimension="conceptual",
            score=score,
            skill=skill_tag or str(llm_result.get("topic_tag") or ""),
            question_id=qid,
            hint_level=hint_now,
            note=(answer or "")[:160],
            source="llm",
            elapsed=_elapsed(row),
        )
        state["skill_graph"] = skill_graph.update_skill(
            graph,
            topic_tag=skill_tag or llm_result.get("topic_tag") or state.get("current_qa_id") or "",
            score_0_100=score,
            evidence=(answer or "")[:160],
            weight=0.55 if score < 25 else 0.35,
        )

        force_coding = _should_enter_coding(row, state)
        # Custom briefings often ask for SQL / Java / DBMS before coding — do not let the
        # model skip the spoken round early while QA time remains.
        if action == "move_to_coding" and not force_coding:
            qa_budget = _qa_budget_seconds(row, state)
            topics = _configured_topics(state)
            cursor = int(state.get("topic_cursor", 0) or 0)
            unfinished = bool(topics) and cursor < max(0, len(topics) - 1)
            if unfinished and _elapsed(row) < int(0.9 * qa_budget):
                action = "next_topic"
                state["force_fresh_question"] = True
            elif _elapsed(row) < int(0.85 * qa_budget):
                action = "next_topic"
                state["force_fresh_question"] = True
        if force_coding or action == "move_to_coding":
            state["coding_after_answer"] = False
            _save_state(row, state)
            # Do not prepend another conceptual question — that is how stems get repeated
            # right as the problem statement appears.
            return _close_qa_then_code(db, row, state)

        # Realtime owns spoken Q&A. Persist scores only — never a coach note or extra stem.
        if llm_result.get("topic_tag"):
            tag = str(llm_result["topic_tag"])[:80]
            if tag.lower() not in _GENERIC_SKILL_TAGS:
                state["current_qa_id"] = tag
        if action == "next_topic":
            _note_topic_turn(state)
            cue = str(llm_result.get("cue") or "").strip()
            # Realtime invents the spoken question, so a resume item only gets asked if it
            # rides along as a cue. Without this the uploaded CV is never touched in duplex.
            if _resume_questions_allowed(row, state) and not _is_resume_track(row.role_track, state):
                planned = _take_planned_resume_question(state)
                if planned:
                    cue = "Resume is uploaded — ask confidently naming the anchor (You mentioned…), never assume: " + planned
            elif _is_resume_track(row.role_track, state):
                plan = list(state.get("resume_plan") or [])
                idx = int(state.get("resume_plan_index") or 0)
                planned = str(plan[idx].get("question") or "").strip() if 0 <= idx < len(plan) else ""
                if planned:
                    cue = "Resume is uploaded — ask confidently naming the anchor (You mentioned…), never assume: " + planned
            if cue:
                state["realtime_cue"] = cue[:300]
        _save_state(row, state)
        return ""

    if llm_client.llm_configured():
        _save_state(row, state)
        return ""

    # Heuristic fallback only without API key.
    keywords = list((node or {}).get("keywords") or [])
    if not keywords:
        bank = {q["id"]: q for q in evaluator.TECH_BANK}
        q = bank.get(state.get("current_qa_id"), evaluator.TECH_BANK[0])
        keywords = q["keywords"]
        score = evaluator.score_keywords(answer, keywords)
    else:
        score = evaluator.score_keywords(answer, keywords)

    state.setdefault("qa_scores", []).append(score)
    state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
    _apply_score_communication(state, answer, score)
    state["qa_index"] = idx + 1
    state["difficulty_ceiling"] = question_graph.adjust_difficulty_ceiling(
        int(state.get("difficulty_ceiling", 2) or 2),
        score,
    )
    evidence_ledger.record(
        state,
        stage="qa",
        dimension="conceptual",
        score=score,
        skill=skill_tag,
        question_id=qid,
        hint_level=hint_now,
        note=(answer or "")[:160],
        source="heuristic",
        elapsed=_elapsed(row),
    )
    state["skill_graph"] = skill_graph.update_skill(
        graph,
        topic_tag=skill_tag,
        score_0_100=score,
        evidence=(answer or "")[:160],
    )
    feedback = _move_on_bridge(state)

    if _should_enter_coding(row, state):
        state["coding_after_answer"] = False
        _save_state(row, state)
        return _close_qa_then_code(db, row, state)

    _save_state(row, state)
    return ""


def _close_qa_then_code(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    """Close conceptual round, then open coding — or wrap when coding is disabled."""
    state["coding_after_answer"] = False
    state.pop("realtime_cue", None)
    if not _include_coding(state):
        return _begin_wrap_no_coding(db, row, state)
    coding = _start_coding_round(db, row, state)
    state["coding_handoff"] = True
    _save_state(row, state)
    agenda_done = _qa_agenda_complete(state)
    lead = (
        "Alright — that wraps the spoken technical questions. "
        if agenda_done
        else "Alright — good. "
    )
    return lead + coding


def _begin_wrap_no_coding(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    """End a conceptual-only interview with brief spoken feedback."""
    row.stage = "wrap"
    state["awaiting"] = "wrap"
    conceptual = float(state.get("score_conceptual") or 0)
    name = (state.get("interviewer_name") or "NexAI").strip() or "NexAI"
    band = "solid" if conceptual >= 70 else ("mixed" if conceptual >= 40 else "needs work")
    _save_state(row, state)
    return (
        f"Thanks — that closes the technical questions for this {name} session. "
        f"Your conceptual round looked {band} overall. "
        "I'll wrap with a short summary on your report. You can end the session when you're ready."
    )


def _start_coding_round(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    prefer = "easy"
    conceptual = float(state.get("score_conceptual", 0) or 0)
    indep = evidence_ledger.independence_score(state)
    if conceptual >= 75 and indep >= 60:
        prefer = "medium"
    if conceptual >= 88 and indep >= 75 and int(state.get("difficulty_ceiling", 2) or 2) >= 4:
        prefer = "medium"
    moodle_pid = int(state.get("moodle_problem_id") or 0)
    title = (state.get("moodle_problem_title") or "the NexPractice problem on your screen").strip()
    if moodle_pid:
        state["awaiting"] = "idea"
        row.stage = "idea"
        state["idea_attempts"] = 0
        state["explain_count"] = 0
        state["idea_started_elapsed"] = _elapsed(row)
        _save_state(row, state)
        return (
            f"Let's move to coding. Open {title} on your screen — the editor stays locked for now. "
            "Walk me through your approach: data structure, main steps, time complexity, and one edge case. "
            "I'll unlock the editor when that plan is solid. I will not solve it for you."
        )

    problem = pick_problem(state.get("used_problems", []), state.get("topics", []), prefer=prefer)
    state["current_problem_id"] = problem["id"]
    state.setdefault("used_problems", []).append(problem["id"])
    state["idea_attempts"] = 0
    state["explain_count"] = 0
    state["current_code"] = problem.get("starter_code", "")
    state["awaiting"] = "idea"
    row.stage = "idea"
    state["idea_started_elapsed"] = _elapsed(row)
    _save_state(row, state)
    return (
        f"Let's move to coding. The problem on your screen is {problem['title']}. "
        "The editor stays locked for now. Walk me through your approach: data structure, "
        "main steps, time complexity, and one edge case. I will unlock the editor once "
        "the idea looks solid. I will not solve it for you."
    )


_SOLUTION_ASK_RE = re.compile(
    r"(?i)\b(solution|give me the (answer|code)|tell me the (answer|solution)|explain the question|"
    r"what is the (problem|question)|which model|chatgpt)\b"
)
_APPROACH_HINTS = (
    "array", "index", "loop", "complex", "o(n)", "hash", "edge", "length",
    "mid", "sort", "scan", "pointer", "sum", "prefix",
    "stack", "queue", "tree", "map", "brut", "recurs", "window", "binary",
    "step", "space", "factor", "divisor", "divis", "kth", "largest",
    "modulo", "sqrt", "prime", "gcd", "count",
)


def _looks_like_coding_approach(text: str) -> bool:
    blob = (text or "").lower()
    words = re.findall(r"[a-z0-9]+", blob)
    if len(words) < 18:
        return False
    hits = sum(1 for k in _APPROACH_HINTS if k in blob)
    return hits >= 2


_READY_TO_CODE_RE = re.compile(
    r"(?i)\b(unlock (the )?editor|let me (start |just )?(code|coding|implement|write)|"
    r"i'?m ready to (code|start|implement)|can i (start |begin )?(code|coding|implement)|"
    r"(let'?s|lets) (code|start coding)|start coding|open the editor)\b"
)
# Max wall-clock the approach discussion may consume before the editor opens anyway.
IDEA_STAGE_MAX_SECONDS = 210


def _idea_seconds(row: SessionRow, state: dict[str, Any]) -> int:
    started = state.get("idea_started_elapsed")
    if started is None:
        return 0
    return max(0, _elapsed(row) - int(started or 0))


def _substantive_idea_answer(text: str) -> bool:
    return len(re.findall(r"[a-z0-9]+", (text or "").lower())) >= 10


def _unlock_editor(
    row: SessionRow,
    state: dict[str, Any],
    note: str,
    *,
    answer: str = "",
) -> str:
    row.stage = "code"
    state["awaiting"] = "code"
    state["editor_just_unlocked"] = True
    # Duplex otherwise strips this line — room.js must speak the unlock.
    state["editor_unlock_speak"] = True
    peak = max(float(state.get("score_idea", 0) or 0), 55.0)
    state["score_idea"] = peak
    qid = str(state.get("current_problem_id") or state.get("moodle_problem_id") or "idea")
    evidence_ledger.record(
        state,
        stage="idea",
        dimension="problem_solving",
        score=peak,
        skill="coding.approach",
        question_id=qid,
        hint_level=0,
        note=(answer or "Approach accepted — editor unlocked")[:160],
        source="unlock",
        elapsed=_elapsed(row),
    )
    explain = round(min(85.0, peak * 0.80 + 10.0), 1)
    state["score_explain"] = max(float(state.get("score_explain", 0) or 0), explain)
    evidence_ledger.record(
        state,
        stage="idea",
        dimension="explanation",
        score=explain,
        skill="coding.explanation",
        question_id=qid,
        hint_level=0,
        note="Pre-coding approach walkthrough",
        source="unlock",
        elapsed=_elapsed(row),
    )
    _save_state(row, state)
    return note


def _handle_idea(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    idea_secs = _idea_seconds(row, state)
    attempts = int(state.get("idea_attempts", 0) or 0)
    if _SOLUTION_ASK_RE.search(answer or ""):
        return (
            "I will not give the solution or read the problem aloud — it is on your screen. "
            "Walk me through your approach: data structure, main steps, time complexity, and one edge case."
        )
    if llm_client.is_weak_answer(answer, min_words=8):
        # Weak/filler never unlocks early — only the 210s timeout does.
        if idea_secs >= IDEA_STAGE_MAX_SECONDS:
            state["score_idea"] = max(float(state.get("score_idea", 0)), 45.0)
            state["idea_hint_level"] = 3
            return _unlock_editor(
                row,
                state,
                "Let's not spend more time on the plan — the editor is unlocked. "
                "Start implementing in NexPractice and I'll ask about your logic as you go.",
            )
        return (
            "Give me a clearer plan first — data structure, main steps, time complexity, and one edge case. "
            "Then I will unlock the editor."
        )

    problem = None
    if state.get("current_problem_id"):
        problem = get_problem(state["current_problem_id"])
    if _substantive_idea_answer(answer):
        state["idea_attempts"] = attempts = attempts + 1
    # "Let me code" is not a plan. Unlock only when they actually described the approach.
    if _READY_TO_CODE_RE.search(answer or "") and _looks_like_coding_approach(answer):
        state["score_idea"] = max(float(state.get("score_idea", 0)), 50.0)
        return _unlock_editor(
            row,
            state,
            "Alright — the editor is unlocked. Implement in NexPractice and run tests when ready. "
            "I can see your code and may ask why you chose that approach.",
        )
    if idea_secs >= IDEA_STAGE_MAX_SECONDS:
        state["score_idea"] = max(float(state.get("score_idea", 0)), 50.0)
        return _unlock_editor(
            row,
            state,
            "That's enough on the plan — the editor is unlocked. Implement in NexPractice and run "
            "tests when ready. I'll ask about your logic while you code.",
        )

    llm_result = llm_client.interviewer_turn(
        stage="idea",
        role_track=row.role_track,
        topics=state.get("topics", []),
        transcript=_recent_transcript(db, row.id),
        student_message=answer,
        context={
            "problem": (
                {"title": problem["title"], "prompt": problem["prompt"]}
                if problem
                else {
                    "title": state.get("moodle_problem_title") or "",
                    "prompt": (state.get("moodle_problem_statement") or "").strip()[:2400],
                    "moodle_problem_id": state.get("moodle_problem_id"),
                    "constraint": (
                        "Ask ONLY about this on-screen NexPractice problem. "
                        "Never invent a different DSA question (no generic duplicates/"
                        "hashmap/indexing unless that is the problem)."
                    ),
                }
            ),
            "idea_attempts": state["idea_attempts"],
            "seconds_remaining": _seconds_remaining(row),
            "resume_text": state.get("resume_text") or "",
            "moodle_problem_id": state.get("moodle_problem_id"),
            "moodle_problem_title": state.get("moodle_problem_title") or "",
        },
    )
    if llm_result:
        score_now = max(float(llm_result["score"]), duplex_score.heuristic_idea_score(answer))
        state["score_idea"] = max(float(state.get("score_idea", 0)), score_now)
        action = llm_result["next_action"]
        unlock = action == "unlock_editor" and score_now >= 50.0 and _looks_like_coding_approach(answer)
        if not unlock and _looks_like_coding_approach(answer) and score_now >= 60.0:
            unlock = True
        idea_hint = evidence_ledger.bump_hint(
            int(state.get("idea_hint_level", 0) or 0),
            reason="probe_idea" if not unlock else "followup",
        )
        if not unlock:
            state["idea_hint_level"] = idea_hint
        evidence_ledger.record(
            state,
            stage="idea",
            dimension="problem_solving",
            score=score_now,
            skill="coding.approach",
            question_id=str(state.get("current_problem_id") or state.get("moodle_problem_id") or "idea"),
            hint_level=0 if unlock else idea_hint,
            note=(answer or "")[:160],
            source="llm",
            elapsed=_elapsed(row),
        )
        if unlock and _substantive_idea_answer(answer):
            explain_seed = round(min(82.0, score_now * 0.78 + 12.0), 1)
            state["score_explain"] = max(float(state.get("score_explain", 0) or 0), explain_seed)
            evidence_ledger.record(
                state,
                stage="idea",
                dimension="explanation",
                score=explain_seed,
                skill="coding.explanation",
                question_id=str(state.get("current_problem_id") or state.get("moodle_problem_id") or "idea"),
                hint_level=0,
                note=(answer or "")[:160],
                source="idea_narration",
                elapsed=_elapsed(row),
            )
        # Hard stops so the approach probe can never become an endless loop.
        if not unlock and attempts >= 3 and idea_secs >= 90:
            unlock = True
            state["idea_hint_level"] = 3
        if unlock:
            return _unlock_editor(
                row,
                state,
                "Your plan is good enough to start — the editor is unlocked. "
                "Implement in the NexPractice IDE and run tests when ready. "
                "I can see your editor and may ask about your logic — I will not give the solution.",
                answer=answer,
            )
        _save_state(row, state)
        # Realtime owns the spoken probe during duplex; keep a short engine note for the report only
        # when it is a real question (not a lecture).
        spoken = str(llm_result.get("reply") or "").strip()
        if spoken and "?" in spoken and "hidden coach" not in spoken.lower():
            return spoken
        return (
            "Keep going on the approach — data structure, main steps, time complexity, and one edge case. "
            "I unlock the editor once that plan is clear."
        )

    if not problem:
        if not _looks_like_coding_approach(answer):
            _save_state(row, state)
            return (
                "The problem is on your screen — I will not read it. "
                "Give me the data structure, main steps, time complexity, and one edge case first."
            )
        row.stage = "code"
        state["awaiting"] = "code"
        state["score_idea"] = max(float(state.get("score_idea", 0)), 55.0)
        state["editor_just_unlocked"] = True
        _save_state(row, state)
        return (
            "Solid enough — the editor is unlocked. Implement in NexPractice and run tests when ready. "
            "I may interrupt you to explain a piece of logic."
        )

    result = evaluator.score_idea(answer, problem)
    state["score_idea"] = max(float(state.get("score_idea", 0)), result["score"])
    unlock = result["accepted"] or attempts >= 3
    if unlock:
        row.stage = "code"
        state["awaiting"] = "code"
        state["editor_just_unlocked"] = True
        _save_state(row, state)
        note = (
            result["feedback"]
            if result["accepted"]
            else "We'll proceed so you still have time to code — keep refining as you go."
        )
        return (
            f"{note}\n\n"
            "The editor is unlocked. Implement in the NexPractice IDE and run tests when ready. "
            "I may ask about the code you write — I will not give the solution."
        )
    _save_state(row, state)
    return result["feedback"] + " Reply with a clearer plan (structure + complexity + edges)."


def _resolve_pending_explain(state: dict[str, Any], row: SessionRow | None = None) -> None:
    """Credit pre-coding approach narration when a live explain prompt was left unanswered."""
    if not state.get("explain_pending"):
        return
    state["explain_pending"] = False
    if float(state.get("score_explain", 0) or 0) > 0:
        return
    proxy = idea_approach_explain_proxy(state)
    if proxy <= 0:
        return
    credit = round(min(75.0, proxy * 0.82), 1)
    state["score_explain"] = max(float(state.get("score_explain", 0) or 0), credit)
    evidence_ledger.record(
        state,
        stage="explain",
        dimension="explanation",
        score=credit,
        skill="coding.explanation",
        question_id=str(state.get("moodle_problem_id") or state.get("current_problem_id") or "explain"),
        hint_level=0,
        note="Approach narration credited (live code walkthrough unanswered)",
        source="idea_carryover",
        elapsed=_elapsed(row) if row is not None else 0,
    )
    if row is not None:
        row.stage = "code"
        state["awaiting"] = "code"


def maybe_interrupt(db: Session, row: SessionRow, state: dict[str, Any], code: str) -> str | None:
    if row.stage != "code":
        return None
    if _should_wrap(row):
        return None
    duplex = bool(state.get("voice_duplex"))
    max_explains = 2 if duplex else 5
    cooldown = 90 if duplex else 40
    if int(state.get("explain_count", 0)) >= max_explains:
        return None
    nontrivial = sum(1 for ln in code.splitlines() if ln.strip() and "pass" not in ln)
    if nontrivial < 3:
        return None
    last_len = int(state.get("interrupt_code_len", 0))
    if last_len and abs(len(code) - last_len) < 24:
        return None
    last_at = float(state.get("last_interrupt_elapsed", 0) or 0)
    if last_at and (_elapsed(row) - last_at) < cooldown:
        return None
    excerpt = evaluator.pick_code_excerpt(code)
    if not excerpt:
        return None
    state["explain_count"] = int(state.get("explain_count", 0)) + 1
    state["interrupt_code_len"] = len(code)
    state["last_interrupt_elapsed"] = _elapsed(row)
    state["realtime_cue"] = (
        "Ask ONE short question about THIS code excerpt only (do not invent a different question): "
        + excerpt.replace("\n", " ")[:220]
    )[:300]
    # Duplex: cue Realtime only — an assistant interrupt turn fights the live mouth.
    if duplex:
        _save_state(row, state)
        return None

    state["explain_excerpt"] = excerpt
    state["explain_pending"] = True
    state["awaiting"] = "explain"
    row.stage = "explain"
    _save_state(row, state)

    llm_result = llm_client.interviewer_turn(
        stage="explain",
        role_track=row.role_track,
        topics=state.get("topics", []),
        transcript=_recent_transcript(db, row.id),
        student_message="(candidate is coding; ask about THEIR code now)",
        context={
            "code_excerpt": excerpt,
            "problem": _current_problem(state) or {"title": state.get("moodle_problem_title")},
            "seconds_remaining": _seconds_remaining(row),
        },
    )
    if llm_result and llm_result.get("reply"):
        return llm_result["reply"]

    cue = (
        "Quick pause — I can see this in your editor: "
        f"{excerpt.replace(chr(10), ' ')}. "
        "Why did you write it this way, and what happens on a duplicate or empty input?"
    )
    state["realtime_cue"] = (
        "Ask about THIS code excerpt only (do not invent a different question): "
        + excerpt.replace("\n", " ")[:220]
    )
    return cue


def _handle_explain(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    state["explain_pending"] = False
    excerpt = state.get("explain_excerpt", "")
    llm_result = llm_client.interviewer_turn(
        stage="explain",
        role_track=row.role_track,
        topics=state.get("topics", []),
        transcript=_recent_transcript(db, row.id),
        student_message=answer,
        context={
            "code_excerpt": excerpt,
            "problem": _current_problem(state),
            "seconds_remaining": _seconds_remaining(row),
        },
    )
    if llm_result:
        score = float(llm_result["score"])
        prev = float(state.get("score_explain", 0))
        state["score_explain"] = score if prev == 0 else (prev + score) / 2
        evidence_ledger.record(
            state,
            stage="explain",
            dimension="explanation",
            score=score,
            skill="coding.explanation",
            question_id=str(state.get("current_problem_id") or "explain"),
            hint_level=0,
            note=(answer or "")[:160],
            source="llm",
            elapsed=_elapsed(row),
        )
        if score < 50:
            state.setdefault("flags", []).append("weak_code_explanation")
        row.stage = "code"
        state["awaiting"] = "code"
        _save_state(row, state)
        return llm_result["reply"]

    score = evaluator.score_explanation(answer, excerpt)
    prev = float(state.get("score_explain", 0))
    state["score_explain"] = score if prev == 0 else (prev + score) / 2
    evidence_ledger.record(
        state,
        stage="explain",
        dimension="explanation",
        score=score,
        skill="coding.explanation",
        question_id=str(state.get("current_problem_id") or "explain"),
        hint_level=0,
        note=(answer or "")[:160],
        source="heuristic",
        elapsed=_elapsed(row),
    )
    if score < 50:
        state.setdefault("flags", []).append("weak_code_explanation")
    row.stage = "code"
    state["awaiting"] = "code"
    _save_state(row, state)
    return (
        "Thanks — keep going in the editor. Run tests when you think you're ready. "
        "Say done if you want to finish the interview."
    )


def handle_message(
    db: Session,
    row: SessionRow,
    message: str,
    *,
    duration_sec: float = 0.0,
) -> dict[str, Any]:
    if row.status != "active":
        return session_view(db, row)
    state = _state(row)
    # Clarification after a green submit should not be cut by the wrap timer mid-sentence.
    if _should_wrap(row) and not state.get("awaiting_coding_clarify"):
        return finish_session(db, row, reason="time_up")
    if state.get("awaiting_coding_clarify") and _seconds_remaining(row) <= 15:
        return finish_session(db, row, reason="time_up")
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)

    text = (message or "").strip()
    if llm_client.is_phantom_transcript(text):
        return session_view(db, row)
    if not text:
        state = _state(row)
        if state.get("awaiting_coding_clarify"):
            reply = "Any clarifying questions before we wrap up, or shall we close?"
            if state.get("voice_duplex"):
                state["realtime_cue"] = reply[:300]
                _save_state(row, state)
                db.commit()
                db.refresh(row)
                return session_view(db, row)
            _add_turn(db, row.id, "wrap", "assistant", reply, {"coding_clarify": True})
            _save_state(row, state)
            db.commit()
            db.refresh(row)
            return session_view(db, row)
        if row.stage in {"qa", "idea"}:
            reply = _reask_current_question(state, reason="empty")
        else:
            reply = "I didn't catch that — please speak your answer clearly."
        _add_turn(db, row.id, row.stage, "assistant", reply)
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    state = _state(row)

    if state.get("awaiting_coding_clarify"):
        return _handle_coding_clarify(db, row, state, text, duration_sec)

    # Wrap-up confirm — duplex "yes" is short and must never be treated as weak filler.
    if state.get("awaiting_end_confirm"):
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text, {"end_confirm": True})
        if _is_yes(text) and not _is_no(text):
            state["awaiting_end_confirm"] = False
            duplex_score.clear_utterance_buffer(state)
            _save_state(row, state)
            return finish_session(db, row, reason="student_ended")
        if _is_no(text):
            state["awaiting_end_confirm"] = False
            state["end_confirm_asks"] = 0
            reply = "Alright — we'll continue."
            _add_turn(db, row.id, row.stage, "assistant", reply, {"end_confirm": True})
            _save_state(row, state)
            db.commit()
            db.refresh(row)
            return session_view(db, row)
        # Second unclear reply after wrap request → end anyway (student already asked to finish).
        asks = int(state.get("end_confirm_asks", 0) or 0) + 1
        state["end_confirm_asks"] = asks
        if asks >= 2 or _is_end_intent(text):
            state["awaiting_end_confirm"] = False
            duplex_score.clear_utterance_buffer(state)
            _save_state(row, state)
            return finish_session(db, row, reason="student_ended")
        reply = "Say yes to end the interview now, or no to keep going."
        _add_turn(db, row.id, row.stage, "assistant", reply, {"end_confirm": True})
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    if _is_end_intent(text):
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text, {"end_intent": True})
        state["awaiting_end_confirm"] = True
        state["end_confirm_asks"] = 0
        duplex_score.clear_utterance_buffer(state)
        reply = "Do you want to end the interview now? Say yes to wrap up, or no to continue."
        _add_turn(db, row.id, row.stage, "assistant", reply, {"end_confirm": True})
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    # Duplex: buffer short STT chips until the utterance is scoreable.
    score_text = duplex_score.take_scoreable_text(state, text, stage=row.stage or "qa")
    if score_text is None:
        # Keep scoring paused, but still show the chip on the recruiter timeline.
        persist = " ".join((text or "").split())
        if duplex_score.word_count(persist) >= 4 or len(persist) >= 20:
            _add_turn(
                db,
                row.id,
                row.stage,
                "student",
                persist,
                {"buffered": True},
            )
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)
    text = score_text

    # Filler during Q&A / idea: rephrase the same question instead of advancing.
    if row.stage in {"qa", "idea"} and llm_client.is_weak_answer(text):
        state = _state(row)
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text, {"weak": True, "voice": state.get("voice_metrics", {}).get("latest")})
        if row.stage == "qa":
            reply = ""
        else:
            floor = duplex_score.heuristic_idea_score(text)
            state["score_idea"] = max(float(state.get("score_idea", 0) or 0), floor * 0.5)
            evidence_ledger.record(
                state,
                stage="idea",
                dimension="problem_solving",
                score=floor * 0.5,
                skill="coding.approach",
                question_id=str(state.get("current_problem_id") or state.get("moodle_problem_id") or "idea"),
                hint_level=int(state.get("idea_hint_level", 0) or 0),
                note=(text or "")[:160],
                source="duplex_weak",
                elapsed=_elapsed(row),
            )
            reply = (
                "That came through incomplete. "
                "Outline the data structure, main steps, and time complexity again."
            )
            state["idea_hint_level"] = evidence_ledger.bump_hint(
                int(state.get("idea_hint_level", 0) or 0),
                reason="followup",
            )
        if reply:
            _add_turn(db, row.id, row.stage, "assistant", reply)
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    _add_turn(db, row.id, row.stage, "student", text)
    state = _state(row)
    _note_voice_and_claims(state, text, duration_sec)
    reply = ""

    if row.stage == "intro":
        # Accept common ready phrases, not only exact "yes".
        ready = re.search(
            r"\b(yes|yeah|yep|ready|start|sure|okay|ok|let'?s go|i'?m ready)\b",
            text.lower(),
        )
        if ready:
            reply = _ensure_specific_question(db, row, state, _begin_qa(db, row, state))
        else:
            reply = "Just say yes when you're ready to begin."
    elif row.stage == "qa":
        reply = _next_qa_or_coding(db, row, state, text)
        if row.stage == "qa" and "?" in (reply or ""):
            reply = _ensure_specific_question(db, row, state, reply)
    elif row.stage == "idea":
        reply = _handle_idea(db, row, state, text)
    elif row.stage == "explain":
        if re.search(r"\b(skip|pass|move on|next)\b", text.lower()):
            _resolve_pending_explain(state, row)
            reply = "No problem — keep coding and run tests when you are ready."
        else:
            reply = _handle_explain(db, row, state, text)
    elif row.stage == "code":
        llm_result = llm_client.interviewer_turn(
            stage="code",
            role_track=row.role_track,
            topics=state.get("topics", []),
            transcript=_recent_transcript(db, row.id),
            student_message=text,
            context=_llm_context(
                row,
                state,
                problem=_current_problem(state),
                code_excerpt=evaluator.pick_code_excerpt(state.get("current_code", "")),
            ),
        )
        if llm_result:
            if llm_result["next_action"] == "wrap_up":
                state["awaiting_end_confirm"] = True
                state["end_confirm_asks"] = 0
                reply = (
                    llm_result["reply"]
                    if "?" in llm_result["reply"]
                    else "Do you want to end the interview now? Say yes to wrap up, or no to keep coding."
                )
            elif any(w in text.lower() for w in ("hint", "give me the answer", "solution")):
                state.setdefault("flags", []).append("asked_for_help")
                reply = llm_result["reply"]
            else:
                reply = llm_result["reply"]
        elif any(w in text.lower() for w in ("hint", "stuck", "help me", "give me the answer", "solution")):
            reply = (
                "I can't give you the solution or walk you through the fix — this is an interview. "
                "Tell me what you are trying next, or keep coding and run your tests. "
                "Say done when you want to wrap up."
            )
            state.setdefault("flags", []).append("asked_for_help")
        else:
            reply = (
                "Understood. Keep working in the editor — I can see your code. "
                "Run tests when ready, and say done when you want to finish."
            )
    else:
        reply = "Session is complete. Open your feedback report for details."

    cleaned = _dedupe_against_history(db, row.id, reply)
    if not cleaned and (reply or "").strip() and "?" in (reply or ""):
        # Prefer a soft re-ask over "don't repeat" when voice answers were unclear.
        cleaned = _reask_current_question(state, reason="unclear") if row.stage in {"qa", "idea"} else (
            "Thanks — stay with that last question and go a bit deeper."
        )
    # Never persist scorer coach notes as spoken interviewer turns.
    if cleaned and re.search(r"(?i)^\s*hidden\s+coach\b", cleaned):
        cleaned = ""
    handoff = bool(state.get("coding_handoff"))
    unlock_speak = bool(state.get("editor_unlock_speak"))
    if (
        state.get("voice_duplex")
        and cleaned
        and not handoff
        and not unlock_speak
        and not state.get("awaiting_end_confirm")
        and row.stage in {"qa", "idea", "code"}
    ):
        cleaned = ""
    if cleaned:
        handoff = bool(state.pop("coding_handoff", None))
        unlock_speak = bool(state.pop("editor_unlock_speak", None))
        meta: dict[str, Any] | None = None
        if handoff or unlock_speak:
            meta = {}
            if handoff:
                meta["coding_handoff"] = True
            if unlock_speak:
                meta["editor_unlock"] = True
        # New interviewer turn closes the prior student utterance buffer.
        duplex_score.clear_utterance_buffer(state)
        _add_turn(
            db,
            row.id,
            row.stage,
            "assistant",
            cleaned,
            meta,
        )
    else:
        # Keep coding_handoff so a later poll can still surface the spoken cutover.
        if not handoff:
            state.pop("coding_handoff", None)
        if not unlock_speak:
            state.pop("editor_unlock_speak", None)
    _save_state(row, state)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def save_snapshot(db: Session, row: SessionRow, code: str, source: str = "autosave") -> dict[str, Any]:
    if row.status != "active":
        return session_view(db, row)
    if _should_wrap(row):
        return finish_session(db, row, reason="time_up")
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)
    state = _state(row)
    state["current_code"] = code
    snap = CodingSnapshotRow(
        session_id=row.id,
        problem_id=state.get("current_problem_id", ""),
        code=code,
        source=source,
    )
    db.add(snap)
    interrupt = maybe_interrupt(db, row, state, code) if source in {"autosave", "run"} else None
    _save_state(row, state)
    if interrupt:
        _add_turn(db, row.id, "explain", "assistant", interrupt, {"interrupt": True})
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def run_code(db: Session, row: SessionRow, code: str, mode: str = "sample") -> tuple[dict[str, Any], dict[str, Any]]:
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row), {
            "ok": False,
            "passed": 0,
            "total": 0,
            "details": [],
            "message": "Session ended",
        }

    if row.stage not in {"code", "explain"}:
        return session_view(db, row), {
            "ok": False,
            "passed": 0,
            "total": 0,
            "details": [],
            "message": "Editor is locked until your approach is accepted.",
        }

    state = _state(row)
    problem = get_problem(state.get("current_problem_id", ""))
    if not problem:
        return session_view(db, row), {
            "ok": False,
            "passed": 0,
            "total": 0,
            "details": [],
            "message": "No active problem",
        }

    tests = problem.get("sample_tests", []) if mode != "hidden" else (
        problem.get("sample_tests", []) + problem.get("hidden_tests", [])
    )
    result = evaluate_code(code, tests)
    state["current_code"] = code
    state["run_count"] = int(state.get("run_count", 0)) + 1
    if not result["ok"]:
        state["run_failures"] = int(state.get("run_failures", 0)) + 1
        if state["run_failures"] >= 4:
            state.setdefault("flags", []).append("many_run_failures")
    else:
        # Coding score blends sample/hidden success.
        ratio = result["passed"] / max(1, result["total"])
        score = 55 + ratio * 45
        state["score_coding"] = max(float(state.get("score_coding", 0)), score)

    snap = CodingSnapshotRow(
        session_id=row.id,
        problem_id=problem["id"],
        code=code,
        source="run",
        run_result_json=_dumps(result),
    )
    db.add(snap)

    msg = result["message"]
    if result["ok"] and mode == "sample":
        # Auto-run hidden quietly for scoring.
        hidden = evaluate_code(code, problem.get("sample_tests", []) + problem.get("hidden_tests", []))
        hr = hidden["passed"] / max(1, hidden["total"])
        state["score_coding"] = max(float(state.get("score_coding", 0)), 50 + hr * 50)
        msg += f". Hidden suite: {hidden['passed']}/{hidden['total']} (used for scoring only)."

    _add_turn(
        db,
        row.id,
        row.stage,
        "system",
        f"Code run ({mode}): {msg}",
        {"run": result},
    )

    interrupt = maybe_interrupt(db, row, state, code)
    _save_state(row, state)
    if interrupt:
        _add_turn(db, row.id, "explain", "assistant", interrupt, {"interrupt": True})
    else:
        _add_turn(
            db,
            row.id,
            row.stage,
            "assistant",
            "Your sample tests passed. Keep refining if you want, or say done to finish."
            if result["ok"]
            else "Some tests failed. Inspect the failing case and continue — I won't provide the fix.",
        )
    db.commit()
    db.refresh(row)
    return session_view(db, row), result


def handle_coding_result(
    db: Session,
    row: SessionRow,
    *,
    passed: int,
    total: int,
    all_passed: bool,
    problem_id: int = 0,
) -> dict[str, Any]:
    """Called when NexPractice submit finishes (Moodle judge)."""
    if row.status != "active":
        return session_view(db, row)
    if _should_wrap(row):
        return finish_session(db, row, reason="time_up")

    state = _state(row)
    passed = max(0, int(passed or 0))
    total = max(0, int(total or 0))
    ratio = passed / max(1, total)
    pid = int(problem_id or state.get("moodle_problem_id") or 0)

    # Duplex: never park a green submit behind explain_pending — credit the pass,
    # soft-resolve the explain debt, then continue. Realtime can still ask about code.
    if all_passed and total > 0 and state.get("explain_pending"):
        _resolve_pending_explain(state, row)
        state.pop("explain_submit_nudge", None)
        if row.stage == "explain":
            row.stage = "code"
            state["awaiting"] = "code"

    if all_passed and total > 0:
        pid_key = str(pid or state.get("current_problem_id") or "moodle")
        solved_ids = [str(x) for x in (state.get("solved_problem_ids") or [])]
        if pid_key in solved_ids:
            _save_state(row, state)
            db.commit()
            db.refresh(row)
            return session_view(db, row)
        solved_ids.append(pid_key)
        state["solved_problem_ids"] = solved_ids
        state["score_coding"] = max(float(state.get("score_coding", 0) or 0), 85.0 + min(15.0, ratio * 15))
        state["problems_solved_count"] = int(state.get("problems_solved_count", 0) or 0) + 1
        evidence_ledger.record(
            state,
            stage="code",
            dimension="coding",
            score=float(state["score_coding"]),
            skill="coding.correctness",
            question_id=str(pid or "moodle"),
            hint_level=int(state.get("idea_hint_level", 0) or 0),
            note=f"All tests passed ({passed}/{total})",
            source="nexpractice",
            elapsed=_elapsed(row),
        )
        _record_explanation_from_coding_success(state, row, passed=passed, total=total)
        solved_n = int(state["problems_solved_count"])
        max_n = int(state.get("max_coding_problems", 2) or 2)
        # Enough time for another round? (~3+ minutes beyond wrap buffer)
        time_ok = _seconds_remaining(row) > (int(get_settings().wrap_seconds or 120) + 180)
        if solved_n < max_n and time_ok:
            state["need_next_problem"] = True
            state["coding_just_passed"] = True
            state["realtime_cue"] = (
                f"All {passed}/{total} tests just passed. Congratulate briefly, then ask ONE short "
                "question about a non-trivial line in their solution — or wrap up if time is low."
            )
            _save_state(row, state)
            _add_turn(
                db,
                row.id,
                row.stage or "code",
                "system",
                f"Coding passed {passed}/{total}; requesting next problem.",
                {"coding_passed": True},
            )
            db.commit()
            db.refresh(row)
            return session_view(db, row)

        # Time left: congratulate + invite clarifications instead of ending abruptly.
        if _seconds_remaining(row) > 40:
            return _begin_post_coding_clarify(
                db, row, state, passed=passed, total=total
            )

        _save_state(row, state)
        spoken = (
            f"All {passed} tests passed — strong finish on coding. "
            "I'll wrap up with brief feedback now."
        )
        _add_turn(db, row.id, row.stage or "code", "assistant", spoken, {"coding_passed": True})
        db.commit()
        return finish_session(db, row, reason="coding_complete")

    # Partial / failed submit — score proportionally, keep coding.
    partial = 35.0 + ratio * 45.0
    state["score_coding"] = max(float(state.get("score_coding", 0) or 0), partial)
    evidence_ledger.record(
        state,
        stage="code",
        dimension="coding",
        score=partial,
        skill="coding.correctness",
        question_id=str(pid or "moodle"),
        hint_level=0,
        note=f"Submit {passed}/{total} passed",
        source="nexpractice",
        elapsed=_elapsed(row),
    )
    _save_state(row, state)
    spoken = (
        f"Submit came back {passed} of {total}. Keep fixing failing cases — I won't give the solution. "
        "Say done if you want to finish."
    )
    _add_turn(db, row.id, row.stage or "code", "assistant", spoken, {"coding_partial": True})
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def assign_moodle_problem(
    db: Session,
    row: SessionRow,
    *,
    problem_id: int,
    problem_title: str = "",
    problem_statement: str = "",
) -> dict[str, Any]:
    """Attach the next NexPractice problem and reopen the approach gate."""
    if row.status != "active":
        return session_view(db, row)
    state = _state(row)
    pid = int(problem_id or 0)
    title = (problem_title or "the next NexPractice problem").strip()[:180]
    if pid <= 0:
        state["need_next_problem"] = False
        _save_state(row, state)
        return finish_session(db, row, reason="no_more_problems")

    used = state.setdefault("used_moodle_problems", [])
    if pid not in used:
        used.append(pid)
    titles = state.setdefault("moodle_problem_titles", [])
    if title and title not in titles:
        titles.append(title)

    state["moodle_problem_id"] = pid
    state["moodle_problem_title"] = title
    state["moodle_problem_statement"] = (problem_statement or "").strip()[:2800]
    state.pop("realtime_cue", None)
    state["need_next_problem"] = False
    state["_remount_ide"] = True
    state["idea_attempts"] = 0
    state["explain_count"] = 0
    state["idea_hint_level"] = 0
    state["explain_pending"] = False
    state.pop("explain_submit_nudge", None)
    state["awaiting"] = "idea"
    row.stage = "idea"
    state["idea_started_elapsed"] = _elapsed(row)
    _save_state(row, state)
    spoken = (
        f"Nice work — all tests passed. Next problem: {title}. "
        "Editor stays locked — walk me through your approach first: "
        "data structure, main steps, complexity, and one edge case."
    )
    _add_turn(db, row.id, "idea", "assistant", spoken, {"next_problem": True, "moodle_problem_id": pid})
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def get_session(db: Session, session_id: str) -> SessionRow | None:
    return db.get(SessionRow, session_id)


def list_reports_for_cm(db: Session, moodle_cm_id: int) -> list[dict[str, Any]]:
    rows = (
        db.query(SessionRow)
        .filter(SessionRow.moodle_cm_id == moodle_cm_id, SessionRow.status == "completed")
        .order_by(SessionRow.ended_at.desc())
        .all()
    )
    return [_report_item(r) for r in rows]


def list_reports_for_user(db: Session, moodle_user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    rows = (
        db.query(SessionRow)
        .filter(SessionRow.moodle_user_id == moodle_user_id, SessionRow.status == "completed")
        .order_by(SessionRow.ended_at.desc())
        .limit(limit)
        .all()
    )
    return [_report_item(r) for r in rows]


def _report_item(r: SessionRow) -> dict[str, Any]:
    report = _loads(r.report_json, {})
    return {
        "session_id": r.id,
        "moodle_user_id": r.moodle_user_id,
        "student_name": r.student_name,
        "role_track": r.role_track,
        "overall_score": r.overall_score,
        "recommendation": r.recommendation,
        "band": report.get("band", ""),
        "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        "dimensions": report.get("dimensions", {}),
        "flags": report.get("flags", []),
    }
