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
from app import llm as llm_client
from app import skills as skill_graph
from app import voice_metrics
from app.config import get_settings
from app.models import CodingSnapshotRow, SessionRow, TurnRow
from app.problems import get_problem, pick_problem, public_problem
from app.report import build_report
from app.runner import evaluate_code


STAGES = ("intro", "qa", "idea", "code", "explain", "wrap", "done")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _loads(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw or "") or default
    except Exception:
        return default


def _dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


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


def _ui_for(row: SessionRow, state: dict[str, Any]) -> dict[str, Any]:
    # Show NexPractice IDE once coding round begins (idea outline + code + explain).
    show_editor = row.stage in {"idea", "code", "explain"} and (
        bool(state.get("moodle_problem_id")) or bool(state.get("current_problem_id"))
    )
    return {
        "show_editor": show_editor,
        "awaiting": state.get("awaiting", "message"),
        "can_run": show_editor and row.stage == "code",
        "can_submit_idea": row.stage == "idea",
        "interrupt_active": row.stage == "explain",
        "moodle_problem_id": int(state.get("moodle_problem_id") or 0),
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
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
    resume = " ".join((resume_text or "").split())[:12000]
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
        "score_communication": 55,
        "current_code": "",
        "resume_text": resume,
        "moodle_problem_id": int(moodle_problem_id or 0),
        "skill_graph": skill_graph.default_graph(role_track),
        "claims": [],
        "voice_metrics": {},
        "voice_metric_samples": [],
    }
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

    greeting = (
        f"Hi {student_name.split()[0] if student_name else 'there'}. "
        f"This is a live voice technical screen for about {duration_minutes} minutes. "
        "I will speak questions and listen while you answer naturally — you do not need to press a mic button. "
        "Later you will code in the NexPractice editor; I can see your work and may ask you to explain. "
        "I will not solve the problem for you. "
        "When you are ready, just say yes."
    )
    _add_turn(db, session_id, "intro", "assistant", greeting)
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


def finish_session(db: Session, row: SessionRow, reason: str = "completed") -> dict[str, Any]:
    state = _state(row)
    if reason == "time_up" and "time_up" not in state.get("flags", []):
        state.setdefault("flags", []).append("time_up")

    # Fill missing dimension scores with conservative defaults from available signal.
    if not state.get("qa_scores") and state.get("score_conceptual", 0) == 0:
        state["score_conceptual"] = 40
    if state.get("score_communication", 0) == 0:
        state["score_communication"] = 50

    turns = [
        {"stage": t.stage, "role": t.role, "content": t.content}
        for t in db.query(TurnRow).filter(TurnRow.session_id == row.id).order_by(TurnRow.seq.asc())
    ]
    report = build_report(state, turns)
    row.status = "completed"
    row.stage = "done"
    row.overall_score = float(report["overall_score"])
    row.recommendation = report["recommendation"]
    row.report_json = _dumps(report)
    row.ended_at = _now()
    _save_state(row, state)

    summary = (
        f"Interview complete ({reason.replace('_', ' ')}).\n\n"
        f"**Overall:** {report['overall_score']:.0f}/100 — {report['band'].replace('_', ' ')}\n"
        f"**Recommendation:** {report['recommendation']}\n\n"
        f"Strengths: {', '.join(report['strengths'])}\n"
        f"Focus areas: {', '.join(report['gaps'])}\n\n"
        + "\n".join(f"• {s}" for s in report["next_steps"])
    )
    _add_turn(db, row.id, "done", "assistant", summary, {"report": True})
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def _recent_transcript(db: Session, session_id: str, limit: int = 20) -> list[dict[str, str]]:
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
    state["score_communication"] = voice_metrics.blend_communication(
        float(state.get("score_communication") or 55),
        metrics,
    )
    for claim in _extract_claims(answer):
        bag = state.setdefault("claims", [])
        if claim not in bag:
            bag.append(claim)
        state["claims"] = bag[-20:]


def _llm_context(row: SessionRow, state: dict[str, Any], **extra: Any) -> dict[str, Any]:
    graph = state.get("skill_graph") or skill_graph.default_graph(row.role_track)
    ctx = {
        "seconds_remaining": _seconds_remaining(row),
        "qa_index": state.get("qa_index", 0),
        "asked_topics": state.get("asked_topics", []),
        "resume_text": state.get("resume_text") or "",
        "moodle_problem_id": state.get("moodle_problem_id"),
        "skill_graph_summary": skill_graph.summarize_for_llm(graph),
        "claims": state.get("claims") or [],
        "voice_metrics_latest": (state.get("voice_metrics") or {}).get("latest"),
    }
    ctx.update(extra)
    return ctx


def _begin_qa(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    row.stage = "qa"
    state["qa_index"] = 0
    state["awaiting"] = "message"
    state["asked_topics"] = []

    dynamic = llm_client.first_question(
        role_track=row.role_track,
        topics=state.get("topics") or _loads(row.topics_json, []),
        resume_text=state.get("resume_text") or "",
    )
    if dynamic:
        state["current_qa_id"] = dynamic.get("topic_tag") or "llm_opening"
        state.setdefault("asked_topics", []).append(state["current_qa_id"])
        state["llm_mode"] = True
        _save_state(row, state)
        return dynamic["reply"]

    if llm_client.llm_configured():
        # Key is set but the live call failed — do NOT fall back to the fixed script.
        err = llm_client.last_error() or "unknown LLM error"
        state["llm_mode"] = False
        _save_state(row, state)
        return (
            "I could not reach the AI interviewer brain just now. "
            f"Please ask an admin to check Railway OPENAI_API_KEY / model. Detail: {err[:180]}. "
            "Say yes again to retry."
        )

    # Offline/dev fallback only when no API key is configured.
    q = evaluator.TECH_BANK[0]
    state["current_qa_id"] = q["id"]
    state["llm_mode"] = False
    _save_state(row, state)
    return (
        "Great. First conceptual question:\n\n"
        f"**{q['question']}**\n\n"
        "Take a clear structured answer."
    )


def _next_qa_or_coding(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    settings = get_settings()
    topics = state.get("topics") or _loads(row.topics_json, [])
    idx = int(state.get("qa_index", 0))
    target = int(settings.qa_target_exchanges or 3)

    if llm_client.is_weak_answer(answer):
        _save_state(row, state)
        return (
            "I need a fuller answer before we move on — take a breath and explain your reasoning "
            "in a few clear sentences. What is the idea, and why does it work?"
        )

    llm_result = llm_client.interviewer_turn(
        stage="qa",
        role_track=row.role_track,
        topics=topics,
        transcript=_recent_transcript(db, row.id),
        student_message=answer,
        context=_llm_context(row, state),
    )

    if llm_result:
        score = float(llm_result["score"])
        action = llm_result["next_action"]
        # Never advance on weak LLM scores that re-ask.
        if action == "followup" and score < 35:
            tag = llm_result.get("topic_tag") or state.get("current_qa_id") or ""
            state["skill_graph"] = skill_graph.update_skill(
                state.get("skill_graph") or skill_graph.default_graph(row.role_track),
                topic_tag=tag,
                score_0_100=score,
                evidence=f"Weak/follow-up on: {(answer or '')[:120]}",
            )
            _save_state(row, state)
            return llm_result["reply"]

        state.setdefault("qa_scores", []).append(score)
        state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
        _apply_score_communication(state, answer, score)
        state["qa_index"] = idx + 1
        state["llm_mode"] = True
        tag = llm_result.get("topic_tag") or ""
        if tag:
            state.setdefault("asked_topics", []).append(tag)
            state["current_qa_id"] = tag
        state["skill_graph"] = skill_graph.update_skill(
            state.get("skill_graph") or skill_graph.default_graph(row.role_track),
            topic_tag=tag or state.get("current_qa_id") or "",
            score_0_100=score,
            evidence=(answer or "")[:160],
        )

        force_coding = state["qa_index"] >= target or _seconds_remaining(row) < 16 * 60
        if action == "move_to_coding" or force_coding:
            _save_state(row, state)
            return llm_result["reply"] + "\n\n" + _start_coding_round(db, row, state)

        _save_state(row, state)
        return llm_result["reply"]

    if llm_client.llm_configured():
        err = llm_client.last_error() or "LLM call failed"
        # Do not bump qa_index on transport failure — ask them to continue.
        _save_state(row, state)
        return (
            "Thanks — I briefly lost the AI connection mid-question. "
            f"({err[:120]}) Please continue your last point, or say done to wrap up."
        )

    # Heuristic fallback only without API key.
    qid = state.get("current_qa_id")
    bank = {q["id"]: q for q in evaluator.TECH_BANK}
    q = bank.get(qid, evaluator.TECH_BANK[0])
    score = evaluator.score_keywords(answer, q["keywords"])
    state.setdefault("qa_scores", []).append(score)
    state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
    _apply_score_communication(state, answer, score)
    state["qa_index"] = idx + 1
    feedback = f"Thanks — noted ({score:.0f}/100)."

    if state["qa_index"] >= target or _seconds_remaining(row) < 18 * 60:
        _save_state(row, state)
        return feedback + "\n\n" + _start_coding_round(db, row, state)

    used = set(state.get("asked_topics", []))
    nxt = next((item for item in evaluator.TECH_BANK if item["id"] not in used), None)
    if not nxt:
        nxt = evaluator.TECH_BANK[state["qa_index"] % len(evaluator.TECH_BANK)]
    state["current_qa_id"] = nxt["id"]
    state.setdefault("asked_topics", []).append(nxt["id"])
    _save_state(row, state)
    return feedback + f"\n\nNext question:\n\n**{nxt['question']}**"


def _start_coding_round(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    prefer = "easy"
    if state.get("score_conceptual", 0) >= 75:
        prefer = "medium"
    moodle_pid = int(state.get("moodle_problem_id") or 0)
    if moodle_pid:
        state["awaiting"] = "idea"
        row.stage = "idea"
        state["idea_attempts"] = 0
        state["explain_count"] = 0
        _save_state(row, state)
        return (
            "Let's move to coding on the NexPractice problem on your screen.\n\n"
            "Before you write code: outline your approach, data structures, complexity, and edge cases. "
            "I will unlock the editor once the idea looks solid. I will not solve it for you."
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
        "Let's move to coding.\n\n"
        f"**Problem: {problem['title']}** ({problem['difficulty']})\n\n"
        f"{problem['prompt']}\n\n"
        "Before you write code: outline your approach, data structures, complexity, and edge cases. "
        "I will unlock the editor once the idea looks solid. I will not solve it for you."
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
        unlock = action == "unlock_editor" or state["idea_attempts"] >= 2
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
    if result["accepted"] or state["idea_attempts"] >= 2:
        row.stage = "code"
        state["awaiting"] = "code"
        _save_state(row, state)
        unlock = (
            result["feedback"]
            if result["accepted"]
            else "We'll proceed so you can show skill in code — keep refining as you go."
        )
        return (
            f"{unlock}\n\n"
            "Editor unlocked. Implement the solution in Python, use Run tests when ready. "
            "I may interrupt you to explain a piece of logic."
        )
    _save_state(row, state)
    return result["feedback"] + " Reply with a clearer plan (structure + complexity + edges)."


def maybe_interrupt(db: Session, row: SessionRow, state: dict[str, Any], code: str) -> str | None:
    if row.stage != "code":
        return None
    if int(state.get("explain_count", 0)) >= 2:
        return None
    if _seconds_remaining(row) < 5 * 60:
        return None
    nontrivial = sum(1 for ln in code.splitlines() if ln.strip() and "pass" not in ln)
    if nontrivial < 6:
        return None
    seed = sum(ord(c) for c in row.id) + int(state.get("explain_count", 0)) * 7
    if seed % 3 != 0 and state.get("explain_count", 0) == 0:
        if int(state.get("run_count", 0)) < 1:
            return None
    excerpt = evaluator.pick_code_excerpt(code)
    if not excerpt:
        return None
    state["explain_excerpt"] = excerpt
    state["explain_count"] = int(state.get("explain_count", 0)) + 1
    state["awaiting"] = "explain"
    row.stage = "explain"
    _save_state(row, state)

    llm_result = llm_client.interviewer_turn(
        stage="explain",
        role_track=row.role_track,
        topics=state.get("topics", []),
        transcript=_recent_transcript(db, row.id),
        student_message="(candidate is coding; interrupt now)",
        context={
            "code_excerpt": excerpt,
            "problem": _current_problem(state),
            "seconds_remaining": _seconds_remaining(row),
        },
    )
    if llm_result and llm_result.get("reply"):
        return llm_result["reply"]

    return (
        "Quick interrupt — I can see this part of your editor:\n"
        f"```python\n{excerpt}\n```\n"
        "Explain what this block is doing and why you chose this approach. "
        "Do not rewrite it for me — just walk me through your reasoning."
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
        if score < 50:
            state.setdefault("flags", []).append("weak_code_explanation")
        row.stage = "code"
        state["awaiting"] = "code"
        _save_state(row, state)
        return llm_result["reply"]

    score = evaluator.score_explanation(answer, excerpt)
    prev = float(state.get("score_explain", 0))
    state["score_explain"] = score if prev == 0 else (prev + score) / 2
    if score < 50:
        state.setdefault("flags", []).append("weak_code_explanation")
    row.stage = "code"
    state["awaiting"] = "code"
    _save_state(row, state)
    return (
        f"Got it ({score:.0f}/100 for explanation clarity). "
        "Continue coding — run tests when you think you're ready. "
        "Say done if you want to finish the interview."
    )


def handle_message(
    db: Session,
    row: SessionRow,
    message: str,
    *,
    duration_sec: float = 0.0,
) -> dict[str, Any]:
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)

    text = (message or "").strip()
    if not text:
        _add_turn(db, row.id, row.stage, "assistant", "I didn't catch that — please speak your answer clearly.")
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    # Filler during Q&A / idea: acknowledge without logging a "scored" student turn advance path.
    if row.stage in {"qa", "idea"} and llm_client.is_weak_answer(text):
        state = _state(row)
        _note_voice_and_claims(state, text, duration_sec)
        _add_turn(db, row.id, row.stage, "student", text, {"weak": True, "voice": state.get("voice_metrics", {}).get("latest")})
        if row.stage == "qa":
            reply = (
                "I need a fuller answer before we move on — take a breath and explain your reasoning "
                "in a few clear sentences."
            )
        else:
            reply = (
                "Give me a clearer plan first — data structure, main steps, time complexity, and one edge case."
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

    if text.lower() in {"done", "finish", "end interview", "end"}:
        return finish_session(db, row, reason="student_ended")

    if row.stage == "intro":
        if any(w in text.lower() for w in ("yes", "ready", "start", "ok", "okay")):
            reply = _begin_qa(db, row, state)
        else:
            reply = "When you're ready, reply **yes** and we'll start the conceptual round."
    elif row.stage == "qa":
        reply = _next_qa_or_coding(db, row, state, text)
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
                _add_turn(db, row.id, row.stage, "assistant", llm_result["reply"])
                _save_state(row, state)
                db.commit()
                return finish_session(db, row, reason="student_ended")
            if any(w in text.lower() for w in ("hint", "give me the answer", "solution")):
                state.setdefault("flags", []).append("asked_for_help")
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

    _add_turn(db, row.id, row.stage, "assistant", reply)
    _save_state(row, state)
    db.commit()
    db.refresh(row)
    return session_view(db, row)


def save_snapshot(db: Session, row: SessionRow, code: str, source: str = "autosave") -> dict[str, Any]:
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
