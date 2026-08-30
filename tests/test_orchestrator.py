"""Basic orchestrator tests (no network)."""

import os

# Use isolated sqlite file for tests.
os.environ["DATABASE_URL"] = "sqlite:///./test_interview.db"
os.environ["SHARED_SECRET"] = "test-secret"

from datetime import datetime, timedelta, timezone

from app.db import SessionLocal, init_db
from app import orchestrator as orch
from app.auth import sign_payload
import time


def _rewind(db, row, seconds_ago: int):
    row.created_at = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    db.commit()
    db.refresh(row)
    return row


def test_flow_idea_and_code():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=1,
            moodle_cm_id=10,
            moodle_instance_id=5,
            student_name="Test Student",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["arrays", "hashmap", "stack"],
        )
        sid = view["session_id"]
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "yes")
        assert view["stage"] == "qa"
        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "Hash maps give average O(1) lookups. Sorted arrays are O(log n) with binary search but use less overhead and keep order. Collisions and extra memory are the main hash trade-offs.",
        )
        assert view["stage"] == "qa"
        row = _rewind(db, orch.get_session(db, sid), 400)
        view = orch.tick_session(db, row)
        assert view["stage"] == "qa"
        row = _rewind(db, orch.get_session(db, sid), 920)
        view = orch.handle_message(
            db,
            row,
            "A stack is LIFO. Valid parentheses and DFS traversal are classic stack problems because order of nesting matters.",
        )
        assert view["stage"] in {"idea", "code"}
        spoken = " ".join(t["content"] for t in view["turns"] if t["role"] == "assistant")
        assert "wraps the technical" in spoken.lower() or "move to coding" in spoken.lower()
        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "I will use a hash map of value to index while scanning once. For each number check if target-num already seen. This is O(n) time and O(n) space. Edge cases: negatives and duplicates.",
        )
        row = orch.get_session(db, sid)
        if row.stage == "idea":
            view = orch.handle_message(
                db,
                row,
                "Use a stack: push opening brackets, pop on matching close using a map. Empty stack at end means valid. O(n) time.",
            )
        row = orch.get_session(db, sid)
        assert row.stage in {"code", "idea", "explain"}
        view = orch.finish_session(db, row, reason="completed")
        assert view["status"] == "completed"
        assert view["report"] is not None
        assert "overall_score" in view["report"]
    finally:
        db.close()


def test_nexai_timed_flow_and_editor_lock():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=2,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Asha Kumar",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["arrays", "hashmap"],
            moodle_problem_id=42,
            moodle_problem_title="Valid Parentheses",
        )
        assert view["duration_minutes"] == 30
        assert view["stage"] == "qa"
        spoken = " ".join(t["content"] for t in view["turns"] if t["role"] == "assistant")
        assert "nexai" in spoken.lower().replace(" ", "")
        assert view["ui"]["editor_locked"] is False

        sid = view["session_id"]
        row = _rewind(db, orch.get_session(db, sid), 920)
        view = orch.tick_session(db, row)
        # Open technical question must be answered before the IDE appears.
        assert view["stage"] == "qa"
        assert view["ui"]["show_editor"] is False

        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "Hash maps give average O(1) lookups. Sorted arrays are O(log n) with binary search but use less memory.",
        )
        assert view["stage"] == "idea"
        assert view["ui"]["show_editor"] is True
        assert view["ui"]["editor_locked"] is True
        joined = " ".join(t["content"] for t in view["turns"]).lower()
        assert "wraps the technical" in joined
        assert "valid parentheses" in joined

        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "I will use a stack: push opening brackets and pop on a matching close. "
            "Time is O(n) and space O(n). Empty string and leftover opens are the edge cases.",
        )
        if view["stage"] == "idea":
            assert view["ui"]["editor_locked"] is True
        else:
            assert view["stage"] in {"code", "explain"}
            assert view["ui"]["editor_locked"] is False

        row = _rewind(db, orch.get_session(db, sid), 1750)
        view = orch.tick_session(db, row)
        assert view["status"] == "completed"
        last = view["turns"][-1]
        assert last["role"] == "assistant"
        assert last.get("meta", {}).get("spoken_wrap") or "nexai" in last["content"].lower()
    finally:
        db.close()


def test_coding_handoff_not_replaced_on_java_topics():
    """Coding cutover text mentions data structures; do not rewrite it as another Java stem."""
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=4,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Ravi",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["java", "dbms"],
            moodle_problem_id=7,
            moodle_problem_title="Two Sum",
        )
        sid = view["session_id"]
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "yes")
        assert view["stage"] == "qa"
        row = _rewind(db, orch.get_session(db, sid), 1000)
        view = orch.handle_message(
            db,
            row,
            "In Java, ArrayList remove during a for-each throws ConcurrentModificationException. "
            "Use an Iterator.remove or collect indices and delete after the loop. "
            "DBMS transactions need isolation so dirty reads do not leak uncommitted rows.",
        )
        assert view["stage"] == "idea"
        assert view["ui"]["show_editor"] is True
        last = [t for t in view["turns"] if t["role"] == "assistant"][-1]
        spoken = last["content"].lower()
        assert "wraps the technical" in spoken or "move to coding" in spoken
        assert last.get("meta", {}).get("coding_handoff") is True
        assert "next question" not in spoken
    finally:
        db.close()


def test_end_asks_permission():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=3,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Dev",
            role_track="sde_intern",
            duration_minutes=17,
            topics=["arrays"],
        )
        sid = view["session_id"]
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "I want to end the interview")
        assert view["status"] == "active"
        assert "yes" in view["turns"][-1]["content"].lower()
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "no, keep going")
        assert view["status"] == "active"
        assert view["stage"] == "qa"
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "end interview")
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "yes")
        assert view["status"] == "completed"
    finally:
        db.close()


def test_coding_just_passed_survives_tick():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=11,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Pat",
            role_track="sde_intern",
            duration_minutes=17,
            topics=["arrays"],
            moodle_problem_id=42,
            moodle_problem_title="Valid Parentheses",
        )
        row = orch.get_session(db, view["session_id"])
        state = orch._state(row)
        state["coding_just_passed"] = True
        orch._save_state(row, state)
        db.commit()
        first = orch.tick_session(db, row)
        assert first["coding_just_passed"] is True
        row = orch.get_session(db, view["session_id"])
        second = orch.tick_session(db, row)
        assert second["coding_just_passed"] is False
    finally:
        db.close()


def test_repeat_pass_same_problem_does_not_end():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=12,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Sam",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["arrays"],
            moodle_problem_id=42,
            moodle_problem_title="Valid Parentheses",
        )
        sid = view["session_id"]
        row = orch.get_session(db, sid)
        state = orch._state(row)
        row.stage = "code"
        state["awaiting"] = "code"
        state["moodle_problem_id"] = 42
        orch._save_state(row, state)
        db.commit()
        first = orch.handle_coding_result(
            db, row, passed=3, total=3, all_passed=True, problem_id=42
        )
        assert first["status"] == "active"
        row = orch.get_session(db, sid)
        second = orch.handle_coding_result(
            db, row, passed=3, total=3, all_passed=True, problem_id=42
        )
        assert second["status"] == "active"
        row = orch.get_session(db, sid)
        state = orch._state(row)
        assert int(state.get("problems_solved_count") or 0) == 1
    finally:
        db.close()


def test_let_me_code_does_not_unlock_without_approach():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=13,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Alex",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["arrays"],
            moodle_problem_id=42,
            moodle_problem_title="Valid Parentheses",
        )
        sid = view["session_id"]
        row = _rewind(db, orch.get_session(db, sid), 920)
        view = orch.handle_message(
            db,
            row,
            "Hash maps give average O(1) lookups. Sorted arrays are O(log n) with binary search but use less memory.",
        )
        assert view["stage"] == "idea"
        assert view["ui"]["editor_locked"] is True
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "ok let me start coding now please")
        assert view["stage"] == "idea"
        assert view["ui"]["editor_locked"] is True
        row = orch.get_session(db, sid)
        view = orch.handle_message(db, row, "yeah um I think a loop maybe")
        assert view["stage"] == "idea"
        assert view["ui"]["editor_locked"] is True
    finally:
        db.close()


def test_log_turn_skips_leftover_qa_during_coding():
    init_db()
    db = SessionLocal()
    try:
        view = orch.start_session(
            db,
            moodle_user_id=14,
            moodle_cm_id=0,
            moodle_instance_id=0,
            student_name="Kim",
            role_track="sde_intern",
            duration_minutes=30,
            topics=["java"],
            moodle_problem_id=7,
            moodle_problem_title="Two Sum",
        )
        sid = view["session_id"]
        row = _rewind(db, orch.get_session(db, sid), 1000)
        view = orch.handle_message(
            db,
            row,
            "In Java, ArrayList remove during a for-each throws ConcurrentModificationException. "
            "Use an Iterator.remove or collect indices and delete after the loop. "
            "DBMS transactions need isolation so dirty reads do not leak uncommitted rows.",
        )
        assert view["stage"] == "idea"
        before = len([t for t in view["turns"] if t["role"] == "assistant"])
        row = orch.get_session(db, sid)
        view = orch.log_assistant_turn(
            db,
            row,
            "In Java, what is the difference between an ArrayList and a LinkedList?",
        )
        after = [t for t in view["turns"] if t["role"] == "assistant"]
        assert len(after) == before
        last = after[-1]
        assert last.get("meta", {}).get("coding_handoff") is True
    finally:
        db.close()


def test_sign():
    ts = int(time.time())
    sig = sign_payload(["start", 1, 2, 3, "sde_intern", 30, ts], secret="test-secret")
    assert len(sig) == 64
