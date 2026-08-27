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
from app.models import CodingSnapshotRow, SessionRow, TurnRow
from app.problems import get_problem, pick_problem, public_problem
from app.report import build_report
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
    r"\b((i (want to|wanna|would like to) (end|stop|finish))|"
    r"(end|stop|finish) (the )?(interview|session)|"
    r"wrap up( the interview)?)\b",
    re.I,
)
YES_RE = re.compile(r"\b(yes|yeah|yep|yup|sure|confirm|please do|go ahead|end it|that's fine|ok)\b", re.I)
NO_RE = re.compile(r"\b(no|nope|continue|keep going|not yet|wait|don't|do not)\b", re.I)


def _is_end_intent(text: str) -> bool:
    t = " ".join((text or "").lower().split()).strip()
    if t in END_EXACT:
        return True
    if len(t.split()) <= 10 and END_PHRASE_RE.search(t):
        return True
    return False


def _is_yes(text: str) -> bool:
    return bool(YES_RE.search(text or ""))


def _is_no(text: str) -> bool:
    return bool(NO_RE.search(text or ""))


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
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()]
    if topic_list:
        parts.append("Focus topics (priority order): " + ", ".join(topic_list[:12]) + ".")
    if avoid_topics:
        parts.append("Do NOT ask about: " + avoid_topics.strip()[:400] + ".")
    parts.append(
        "Invent fresh questions aligned to these rules. Do not default to generic hash-map / Big-O openers "
        "unless topics or briefing explicitly include them."
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

    Default ~30% Q&A / ~5% wrap (rest coding). Custom interviewers may set
    qa_seconds explicitly so a 30-minute screen can be e.g. 10 min technical + coding.
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
        qa_share = float(settings.qa_share or 0.30)
        qa = int(settings.qa_seconds) if int(settings.qa_seconds or 0) > 0 else int(total * qa_share)
    return qa, wrap


def _include_coding(state: dict[str, Any]) -> bool:
    return bool(state.get("include_coding", True))


def _should_enter_coding(row: SessionRow, state: dict[str, Any]) -> bool:
    if not _include_coding(state):
        return False
    qa_secs, _wrap = _phase_bounds(row, state)
    # Conceptual-only profiles: stretch Q&A until wrap window.
    return _elapsed(row) >= qa_secs or bool(state.get("coding_after_answer"))


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
        "remount_ide": bool(state.get("_remount_ide")),
        "need_next_problem": bool(state.get("need_next_problem")),
        "used_moodle_problems": list(state.get("used_moodle_problems") or []),
        "problems_solved_count": int(state.get("problems_solved_count", 0) or 0),
    }


def _current_problem(state: dict[str, Any]) -> dict[str, Any] | None:
    pid = state.get("current_problem_id")
    if not pid:
        return None
    p = get_problem(pid)
    return public_problem(p) if p else None


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
    scores = {
        "conceptual": state.get("score_conceptual", 0),
        "idea": state.get("score_idea", 0),
        "coding": state.get("score_coding", 0),
        "explain": state.get("score_explain", 0),
        "communication": state.get("score_communication", 0),
        "independence": evidence_ledger.independence_score(state),
        "problems_solved": int(state.get("problems_solved_count", 0) or 0),
        "qa_answers": len(state.get("qa_scores") or []),
    }
    spoken = llm_client.wrap_speech(
        student_name=row.student_name,
        scores=scores,
        flags=list(state.get("flags") or []),
        evidence_tail=(state.get("evidence") or [])[-8:],
        problem_titles=list(state.get("moodle_problem_titles") or []),
    )
    if spoken:
        return spoken
    bits = []
    if scores["qa_answers"] == 0 and scores["problems_solved"] == 0:
        return (
            f"{first}, this is NexAI wrapping up. You didn't submit spoken answers or a passing solution "
            "this round, so scores stay low. Come back when you can talk through one concept and finish "
            "one timed problem. Thanks for trying — your report is next."
        )
    if scores["conceptual"] >= 70:
        bits.append("your conceptual answers were solid")
    elif scores["conceptual"] >= 40:
        bits.append("your conceptual answers were mixed and need more depth")
    else:
        bits.append("conceptual answers were missing or too thin")
    if scores["idea"] >= 60:
        bits.append("the coding approach was reasonable")
    else:
        bits.append("the approach before coding was thin")
    if scores["coding"] >= 70:
        bits.append(f"you cleared {scores['problems_solved'] or 1} coding problem(s)")
    elif scores["coding"] >= 40:
        bits.append("coding was partial — keep pushing tests to green")
    else:
        bits.append("keep practicing timed coding to a full pass")
    return (
        f"{first}, this is NexAI wrapping up. "
        + ", ".join(bits)
        + ". Thanks for the session — you'll see a short report next."
    )


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


def tick_session(db: Session, row: SessionRow) -> dict[str, Any]:
    """Apply timed phase changes. Never open coding while a technical question is unanswered."""
    if row.status != "active":
        return session_view(db, row)
    if _should_wrap(row):
        return finish_session(db, row, reason="time_up")
    state = _state(row)
    if row.stage == "qa" and _elapsed(row) >= _qa_budget_seconds(row, state):
        if _awaiting_student_reply(db, row):
            if not state.get("coding_after_answer"):
                state["coding_after_answer"] = True
                _save_state(row, state)
                db.commit()
                db.refresh(row)
            return session_view(db, row)
        spoken = _close_qa_then_code(db, row, state)
        _add_turn(db, row.id, row.stage, "assistant", spoken)
        db.commit()
        db.refresh(row)
        return session_view(db, row)
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)
    return session_view(db, row)


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
    """Resume-anchored questions only on the resume track or when faculty asked for them."""
    if _is_resume_track(row.role_track, state):
        return True
    briefing = (state.get("interviewer_briefing") or "").lower()
    return any(k in briefing for k in ("resume", "cv ", " cv", "their projects", "past projects"))


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

    state["current_question_id"] = ""
    state["followup_index"] = 0
    state["current_hint_level"] = 0
    _note_topic_turn(state)
    if generated and generated.get("reply"):
        if generated.get("topic_tag"):
            state["current_qa_id"] = str(generated["topic_tag"])[:80]
        elif focus:
            state["current_qa_id"] = focus[:80]
        _save_state(row, state)
        return str(generated["reply"]).strip()

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
        # Two failed attempts at one idea: change topic rather than argue.
        if repairs >= 2:
            return _next_topic_turn(
                db,
                row,
                state,
                bridge="Fair enough — let me ask something clearer.",
            )
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
        state["current_hint_level"] = evidence_ledger.bump_hint(hint_now, reason="followup")
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
        if nxt:
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
    observer = None
    try:
        observer = llm_client.observe_answer(
            stage="qa",
            role_track=row.role_track,
            topics=topics,
            last_question=last_q,
            student_message=answer,
            code_excerpt=str(state.get("current_code") or "")[:1200],
            difficulty=str(state.get("difficulty") or "intermediate"),
            asked_questions=_asked_question_texts(db, row),
            probes_on_topic=int(state.get("followup_index", 0) or 0),
            interviewer_style=str(state.get("interviewer_style") or "friendly"),
        )
    except Exception:
        observer = None
    if observer:
        state["last_observer"] = observer
        trail = state.setdefault("observer_trail", [])
        trail.append({
            "qa_index": idx,
            "depth": observer.get("depth"),
            "score_hint": observer.get("score_hint"),
            "gaps": observer.get("gaps") or [],
            "probe": (observer.get("probe") or "")[:160],
        })
        state["observer_trail"] = trail[-24:]

    ctx = _llm_context(row, state)
    ctx["asked_questions"] = _asked_question_texts(db, row)
    if observer:
        ctx["observer"] = observer

    llm_result = llm_client.interviewer_turn(
        stage="qa",
        role_track=row.role_track,
        topics=topics,
        transcript=_recent_transcript(db, row.id),
        student_message=answer,
        context=ctx,
    )

    if llm_result:
        # Prefer observer score hint when the model is overly generous on thin answers.
        model_score = float(llm_result["score"])
        if observer and observer.get("depth") in {"superficial", "partial"}:
            model_score = min(model_score, float(observer.get("score_hint", model_score)))
        score = llm_client.clamp_answer_score(answer, model_score)
        action = llm_result["next_action"]
        if observer and observer.get("next_action_hint") == "followup" and action == "next_topic":
            # Observer says still shallow — force a probe unless depth budget exhausted.
            max_fu = int(state.get("max_followups_per_question", 2) or 2)
            if int(state.get("followup_index", 0) or 0) < max_fu:
                action = "followup"
                probe = str(observer.get("probe") or "").strip()
                ack = str(observer.get("ack") or "").strip()
                if not ack or re.match(r"(?i)^(got it|understood)\.?$", ack):
                    ack = llm_client.default_ack(str(state.get("interviewer_style") or "friendly"))
                if probe and "?" in probe:
                    llm_result["reply"] = f"{ack} {probe}".strip()
                llm_result["next_action"] = "followup"
        reported_hint = int(llm_result.get("hint_level", hint_now) or hint_now)

        # Off-topic / very weak: never advance — force a rephrase follow-up
        # (unless follow-up depth budget is exhausted → move on).
        max_fu = int(state.get("max_followups_per_question", 2) or 2)
        fu_count = int(state.get("followup_index", 0) or 0)
        dynamic_ok = (
            bool(state.get("interviewer_briefing"))
            or bool(state.get("moodle_interviewer_id"))
            or _is_resume_track(row.role_track, state)
        )
        if action != "followup" and (
            score < 40
            or llm_client.looks_incomplete_answer(answer)
            or llm_client.is_weak_answer(answer)
        ):
            if fu_count < max_fu:
                action = "followup"
            else:
                action = "next_topic"
                # Bridge only — the real question is generated below so the
                # candidate never hears a bare "let's move on".
                llm_result["reply"] = _move_on_bridge(state)
                llm_result["next_action"] = "next_topic"
                state["force_fresh_question"] = True

        # Never advance on weak LLM scores that re-ask.
        if action == "followup" and score < 55:
            if fu_count >= max_fu:
                # Depth budget used — accept score and move topic instead of looping.
                action = "next_topic"
                llm_result["reply"] = _move_on_bridge(state)
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
                reply = (llm_result.get("reply") or "").strip()
                # Bank-stem injection only for non-custom bank mode — custom / briefing
                # interviews must stay on the LLM (and faculty rules), not DSA stems.
                if (not dynamic_ok) and node and new_hint >= 1:
                    stem = question_graph.spoken_prompt(
                        node,
                        hint_level=new_hint,
                        followup_index=int(state.get("followup_index", 0) or 0),
                    )
                    if "?" not in reply:
                        lead = reply.rstrip() or "Let me rephrase."
                        reply = f"{lead} {stem}".strip()
                if reply and (
                    llm_client.is_vague_question(reply)
                    or llm_client.is_off_lock_dsa(reply, _configured_topics(state))
                ):
                    # A generic probe wastes the turn — change topic with a real question.
                    return _next_topic_turn(db, row, state)
                if not reply:
                    reply = "I didn't catch a clear answer — could you rephrase that in one sentence?"
                if llm_result.get("topic_tag"):
                    state["current_qa_id"] = str(llm_result["topic_tag"])[:80]
                _save_state(row, state)
                return reply

        if action == "next_topic":
            state["followup_index"] = 0
            state["current_hint_level"] = 0

        state.setdefault("qa_scores", []).append(score)
        state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
        _apply_score_communication(state, answer, score)
        state["qa_index"] = idx + 1
        if _is_resume_track(row.role_track, state) and action != "followup":
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
        if force_coding or action == "move_to_coding":
            state["coding_after_answer"] = False
            _save_state(row, state)
            # Do not prepend another conceptual question — that is how stems get repeated
            # right as the problem statement appears.
            return _close_qa_then_code(db, row, state)

        # Advance along the curated graph (weakest-skill policy).
        briefing = state.get("interviewer_briefing") or ""
        resume_only = _is_resume_track(row.role_track, state)
        dynamic_ok = bool(briefing) or bool(state.get("moodle_interviewer_id")) or resume_only
        # Custom / briefed interviews with a topic list never touch the DSA bank —
        # that is what pulled sessions into hash-map / complexity questions.
        topic_locked = _topic_locked(row, state)
        force_fresh = bool(state.pop("force_fresh_question", False))
        model_reply_raw = (llm_result.get("reply") or "").strip()
        if force_fresh or (
            action == "next_topic"
            and model_reply_raw
            and (
                llm_client.is_vague_question(model_reply_raw)
                or llm_client.is_off_lock_dsa(model_reply_raw, _configured_topics(state))
            )
        ):
            # The engine (not the model) decided to change topic, or the model's
            # question was too generic to speak — generate a real one.
            bridge = model_reply_raw if force_fresh else ""
            return _next_topic_turn(db, row, state, bridge=bridge)
        if action == "next_topic":
            _note_topic_turn(state)
        nxt = None if (resume_only or topic_locked) else question_graph.pick_next(
            role_track=row.role_track,
            graph=state["skill_graph"],
            asked_ids=list(state.get("asked_question_ids") or []),
            difficulty_ceiling=int(state.get("difficulty_ceiling", 2) or 2),
            topics=topics,
            briefing=briefing,
            # Soft bank hint only; do not force snippet nodes every turn.
            prefer_format=None,
        )
        model_reply = (llm_result.get("reply") or "").strip()
        # With a faculty briefing / custom interviewer, trust the LLM's next question
        # so sessions aren't locked to the same bank stems. Do not activate a bank
        # node we are not going to speak — that caused the next turn to re-ask it.
        if dynamic_ok and model_reply and "?" in model_reply:
            if llm_result.get("topic_tag"):
                state["current_qa_id"] = str(llm_result["topic_tag"])[:80]
            state["current_question_id"] = ""
            _save_state(row, state)
            return model_reply
        if nxt:
            spoken = question_graph.spoken_prompt(nxt, hint_level=0)
            if model_reply and _is_similar_q(model_reply, spoken):
                _activate_question(state, nxt)
                _save_state(row, state)
                return spoken
            if "?" in model_reply and len(model_reply) < 280:
                _activate_question(state, nxt)
                _save_state(row, state)
                return model_reply
            _activate_question(state, nxt)
            bridge = model_reply.split("?")[0].strip() if model_reply else "Thanks."
            if "?" in (model_reply or ""):
                # Model already asked something — don't also speak the bank stem.
                out = model_reply
            else:
                if bridge and not bridge.endswith((".", "!", "?")):
                    bridge += "."
                out = f"{bridge} {spoken}".strip()
            _save_state(row, state)
            return out

        _save_state(row, state)
        reply = (llm_result.get("reply") or "").strip()
        if "?" not in reply:
            # Never trail off without a question — ask a real one on the next topic.
            return _next_topic_turn(db, row, state, bridge=reply)
        return reply

    if llm_client.llm_configured():
        err = llm_client.last_error() or "LLM call failed"
        _save_state(row, state)
        return (
            "Thanks — I briefly lost the AI connection mid-question. "
            "Please continue your last point, or say done to wrap up."
        )

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

    if _topic_locked(row, state):
        _save_state(row, state)
        return _ensure_specific_question(db, row, state, feedback)

    nxt = question_graph.pick_next(
        role_track=row.role_track,
        graph=state["skill_graph"],
        asked_ids=list(state.get("asked_question_ids") or []),
        difficulty_ceiling=int(state.get("difficulty_ceiling", 2) or 2),
        topics=state.get("topics") or _loads(row.topics_json, []),
        briefing=state.get("interviewer_briefing") or "",
    )
    if nxt:
        _activate_question(state, nxt)
        _save_state(row, state)
        return feedback + f"\n\nNext question:\n\n**{nxt['stem']}**"

    used = set(state.get("asked_topics", []))
    bank_nxt = next((item for item in evaluator.TECH_BANK if item["id"] not in used), None)
    if not bank_nxt:
        bank_nxt = evaluator.TECH_BANK[state["qa_index"] % len(evaluator.TECH_BANK)]
    state["current_qa_id"] = bank_nxt["id"]
    state.setdefault("asked_topics", []).append(bank_nxt["id"])
    _save_state(row, state)
    return feedback + f"\n\nNext question:\n\n**{bank_nxt['question']}**"


def _close_qa_then_code(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    """Close conceptual round, then open coding — or wrap when coding is disabled."""
    state["coding_after_answer"] = False
    if not _include_coding(state):
        return _begin_wrap_no_coding(db, row, state)
    coding = _start_coding_round(db, row, state)
    return (
        "Thanks — that wraps the technical questions. "
        + coding
    )


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
    _save_state(row, state)
    return (
        f"Let's move to coding. The problem on your screen is {problem['title']}. "
        "The editor stays locked for now. Walk me through your approach: data structure, "
        "main steps, time complexity, and one edge case. I will unlock the editor once "
        "the idea looks solid. I will not solve it for you."
    )


def _handle_idea(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    if llm_client.is_weak_answer(answer, min_words=8):
        return (
            "Give me a clearer plan first — data structure, main steps, time complexity, and one edge case. "
            "Then I will unlock the editor."
        )

    problem = None
    if state.get("current_problem_id"):
        problem = get_problem(state["current_problem_id"])
    state["idea_attempts"] = int(state.get("idea_attempts", 0)) + 1

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
                else {"moodle_problem_id": state.get("moodle_problem_id")}
            ),
            "idea_attempts": state["idea_attempts"],
            "seconds_remaining": _seconds_remaining(row),
            "resume_text": state.get("resume_text") or "",
            "moodle_problem_id": state.get("moodle_problem_id"),
        },
    )
    if llm_result:
        state["score_idea"] = max(float(state.get("score_idea", 0)), float(llm_result["score"]))
        action = llm_result["next_action"]
        unlock = action == "unlock_editor"
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
            score=float(llm_result["score"]),
            skill="coding.approach",
            question_id=str(state.get("current_problem_id") or state.get("moodle_problem_id") or "idea"),
            hint_level=0 if unlock else idea_hint,
            note=(answer or "")[:160],
            source="llm",
            elapsed=_elapsed(row),
        )
        # Don't unlock just because they tried twice — only if NexAI is satisfied,
        # or the coding window is almost gone.
        if not unlock and int(state.get("idea_attempts", 0)) >= 1 and _seconds_remaining(row) <= (
            int(get_settings().wrap_seconds or 120) + 180
        ):
            unlock = True
        if not unlock and int(state.get("idea_attempts", 0)) >= 5:
            unlock = True
            state["idea_hint_level"] = 3
        if unlock:
            row.stage = "code"
            state["awaiting"] = "code"
            _save_state(row, state)
            return (
                llm_result["reply"]
                + "\n\nEditor unlocked. Implement in the NexPractice IDE and run tests when ready. "
                "I can see your editor and may ask about your logic — I will not give the solution."
            )
        _save_state(row, state)
        return llm_result["reply"]

    if not problem:
        row.stage = "code"
        state["awaiting"] = "code"
        state["score_idea"] = max(float(state.get("score_idea", 0)), 55.0)
        _save_state(row, state)
        return (
            "Solid enough — editor unlocked. Implement in NexPractice and run tests when ready. "
            "I may interrupt you to explain a piece of logic."
        )

    result = evaluator.score_idea(answer, problem)
    state["score_idea"] = max(float(state.get("score_idea", 0)), result["score"])
    unlock = result["accepted"] or int(state.get("idea_attempts", 0)) >= 5
    if unlock:
        row.stage = "code"
        state["awaiting"] = "code"
        _save_state(row, state)
        note = (
            result["feedback"]
            if result["accepted"]
            else "We'll proceed so you still have time to code — keep refining as you go."
        )
        return (
            f"{note}\n\n"
            "Editor unlocked. Implement in the NexPractice IDE and run tests when ready. "
            "I may ask about the code you write — I will not give the solution."
        )
    _save_state(row, state)
    return result["feedback"] + " Reply with a clearer plan (structure + complexity + edges)."


def maybe_interrupt(db: Session, row: SessionRow, state: dict[str, Any], code: str) -> str | None:
    if row.stage != "code":
        return None
    if _should_wrap(row):
        return None
    if int(state.get("explain_count", 0)) >= 5:
        return None
    nontrivial = sum(1 for ln in code.splitlines() if ln.strip() and "pass" not in ln)
    if nontrivial < 3:
        return None
    last_len = int(state.get("interrupt_code_len", 0))
    if last_len and abs(len(code) - last_len) < 24:
        return None
    last_at = float(state.get("last_interrupt_elapsed", 0) or 0)
    if last_at and (_elapsed(row) - last_at) < 40:
        return None
    excerpt = evaluator.pick_code_excerpt(code)
    if not excerpt:
        return None
    state["explain_excerpt"] = excerpt
    state["explain_count"] = int(state.get("explain_count", 0)) + 1
    state["interrupt_code_len"] = len(code)
    state["last_interrupt_elapsed"] = _elapsed(row)
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

    return (
        "Quick pause — I can see this in your editor: "
        f"{excerpt.replace(chr(10), ' ')}. "
        "Why did you write it this way, and what happens on a duplicate or empty input?"
    )


def _handle_explain(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
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
    if _should_wrap(row):
        return finish_session(db, row, reason="time_up")
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)

    text = (message or "").strip()
    if not text:
        state = _state(row)
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

    if state.get("awaiting_end_confirm"):
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text)
        if _is_yes(text) and not _is_no(text):
            state["awaiting_end_confirm"] = False
            _save_state(row, state)
            return finish_session(db, row, reason="student_ended")
        if _is_no(text):
            state["awaiting_end_confirm"] = False
            reply = "Alright — we'll continue."
            _add_turn(db, row.id, row.stage, "assistant", reply)
            _save_state(row, state)
            db.commit()
            db.refresh(row)
            return session_view(db, row)
        reply = "Say yes to end the interview now, or no to keep going."
        _add_turn(db, row.id, row.stage, "assistant", reply)
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    if _is_end_intent(text):
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text)
        state["awaiting_end_confirm"] = True
        reply = "Do you want to end the interview now? Say yes to wrap up, or no to continue."
        _add_turn(db, row.id, row.stage, "assistant", reply)
        _save_state(row, state)
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    # Filler during Q&A / idea: rephrase the same question instead of advancing.
    if row.stage in {"qa", "idea"} and llm_client.is_weak_answer(text):
        state = _state(row)
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text, {"weak": True, "voice": state.get("voice_metrics", {}).get("latest")})
        if row.stage == "qa":
            reply = _reask_current_question(state, reason="weak")
        else:
            reply = (
                "That came through incomplete. "
                "Outline the data structure, main steps, and time complexity again."
            )
            state["idea_hint_level"] = evidence_ledger.bump_hint(
                int(state.get("idea_hint_level", 0) or 0),
                reason="followup",
            )
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
        reply = _ensure_specific_question(
            db, row, state, _next_qa_or_coding(db, row, state, text)
        )
    elif row.stage == "idea":
        reply = _handle_idea(db, row, state, text)
    elif row.stage == "explain":
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
    if cleaned:
        _add_turn(db, row.id, row.stage, "assistant", cleaned)
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

    if all_passed and total > 0:
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
        solved_n = int(state["problems_solved_count"])
        max_n = int(state.get("max_coding_problems", 2) or 2)
        # Enough time for another round? (~3+ minutes beyond wrap buffer)
        time_ok = _seconds_remaining(row) > (int(get_settings().wrap_seconds or 120) + 180)
        if solved_n < max_n and time_ok:
            state["need_next_problem"] = True
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
    state["need_next_problem"] = False
    state["_remount_ide"] = True
    state["idea_attempts"] = 0
    state["explain_count"] = 0
    state["idea_hint_level"] = 0
    state["awaiting"] = "idea"
    row.stage = "idea"
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
