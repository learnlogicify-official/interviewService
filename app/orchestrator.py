"""Interview state machine orchestrator."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app import evaluator
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
    show_editor = row.stage in {"code", "explain"} and bool(state.get("current_problem_id"))
    return {
        "show_editor": show_editor,
        "awaiting": state.get("awaiting", "message"),
        "can_run": show_editor and row.stage == "code",
        "can_submit_idea": row.stage == "idea",
        "interrupt_active": row.stage == "explain",
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
        "ui": _ui_for(row, state),
        "scores": {
            "conceptual": state.get("score_conceptual", 0),
            "idea": state.get("score_idea", 0),
            "coding": state.get("score_coding", 0),
            "explain": state.get("score_explain", 0),
            "communication": state.get("score_communication", 0),
            "overall": row.overall_score,
        },
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
) -> dict[str, Any]:
    session_id = uuid.uuid4().hex
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
        f"Hi {student_name.split()[0] if student_name else 'there'} — I'm your AI technical interviewer "
        f"for a campus SDE-style screen ({duration_minutes} minutes).\n\n"
        "We'll do a short conceptual round, then a coding problem where I'll ask for your approach "
        "before you code. While you code I may pause you to explain a piece of logic.\n\n"
        "Ready? Reply **yes** to begin."
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


def _begin_qa(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    row.stage = "qa"
    state["qa_index"] = 0
    q = evaluator.TECH_BANK[0]
    state["current_qa_id"] = q["id"]
    state["awaiting"] = "message"
    _save_state(row, state)
    return (
        "Great. First conceptual question:\n\n"
        f"**{q['question']}**\n\n"
        "Take a clear structured answer (4–8 sentences is ideal)."
    )


def _next_qa_or_coding(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    qid = state.get("current_qa_id")
    bank = {q["id"]: q for q in evaluator.TECH_BANK}
    q = bank.get(qid, evaluator.TECH_BANK[0])
    score = evaluator.score_keywords(answer, q["keywords"])
    state.setdefault("qa_scores", []).append(score)
    state["score_conceptual"] = sum(state["qa_scores"]) / len(state["qa_scores"])
    words = len(answer.split())
    state["score_communication"] = min(
        100.0,
        45 + min(35, words / 2) + (10 if score >= 60 else 0),
    )

    idx = int(state.get("qa_index", 0)) + 1
    state["qa_index"] = idx
    feedback = f"Thanks — I've noted that ({score:.0f}/100 for concept coverage)."

    # 2 conceptual questions then coding, or advance early if time low.
    if idx >= 2 or _seconds_remaining(row) < 18 * 60:
        return feedback + "\n\n" + _start_coding_round(db, row, state)

    nq = evaluator.TECH_BANK[idx % len(evaluator.TECH_BANK)]
    state["current_qa_id"] = nq["id"]
    _save_state(row, state)
    return feedback + f"\n\nNext question:\n\n**{nq['question']}**"


def _start_coding_round(db: Session, row: SessionRow, state: dict[str, Any]) -> str:
    prefer = "easy"
    if state.get("score_conceptual", 0) >= 75:
        prefer = "medium"
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
        "I will unlock the editor once the idea looks solid."
    )


def _handle_idea(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    problem = get_problem(state["current_problem_id"])
    assert problem
    state["idea_attempts"] = int(state.get("idea_attempts", 0)) + 1
    result = evaluator.score_idea(answer, problem)
    state["score_idea"] = max(float(state.get("score_idea", 0)), result["score"])
    if result["accepted"] or state["idea_attempts"] >= 2:
        row.stage = "code"
        state["awaiting"] = "code"
        _save_state(row, state)
        unlock = (
            result["feedback"]
            if result["accepted"]
            else "We'll proceed so you can show skill in code — keep refining the approach as you go."
        )
        return (
            f"{unlock}\n\n"
            "Editor unlocked. Implement the solution in Python, use **Run tests** when ready. "
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
    # Interrupt after some code and at least one autosave/run.
    nontrivial = sum(1 for ln in code.splitlines() if ln.strip() and "pass" not in ln)
    if nontrivial < 6:
        return None
    # Pseudo-random based on session id + explain count.
    seed = sum(ord(c) for c in row.id) + int(state.get("explain_count", 0)) * 7
    if seed % 3 != 0 and state.get("explain_count", 0) == 0:
        # First interrupt a bit later unless forced by high run count.
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
    return (
        "Quick interrupt — pause coding for a moment.\n\n"
        "Look at this part of your code:\n"
        f"```python\n{excerpt}\n```\n"
        "Explain what it does and why you wrote it this way."
    )


def _handle_explain(db: Session, row: SessionRow, state: dict[str, Any], answer: str) -> str:
    excerpt = state.get("explain_excerpt", "")
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
        "Say **done** if you want to finish the interview."
    )


def handle_message(db: Session, row: SessionRow, message: str) -> dict[str, Any]:
    if _expire_if_needed(db, row):
        db.refresh(row)
        return session_view(db, row)

    text = (message or "").strip()
    if not text:
        _add_turn(db, row.id, row.stage, "assistant", "I didn't catch that — please type your answer.")
        db.commit()
        db.refresh(row)
        return session_view(db, row)

    _add_turn(db, row.id, row.stage, "student", text)
    state = _state(row)
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
        # Free-form during coding: treat as clarification / progress note.
        if "stuck" in text.lower() or "hint" in text.lower():
            reply = (
                "Hint: name the invariant you need, pick the simplest correct structure first, "
                "then optimize. Try a sample case on paper, then code."
            )
        else:
            reply = (
                "Keep going in the editor. Use **Run tests** to check samples. "
                "Say **done** to wrap up the interview."
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
            "Nice — tests look good. You can refine edge cases or say **done** to finish."
            if result["ok"]
            else "Some tests failed. Read the failing case, fix, and run again. Ask for a **hint** if stuck.",
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
    out = []
    for r in rows:
        report = _loads(r.report_json, {})
        out.append(
            {
                "session_id": r.id,
                "moodle_user_id": r.moodle_user_id,
                "student_name": r.student_name,
                "overall_score": r.overall_score,
                "recommendation": r.recommendation,
                "band": report.get("band", ""),
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                "dimensions": report.get("dimensions", {}),
                "flags": report.get("flags", []),
            }
        )
    return out
