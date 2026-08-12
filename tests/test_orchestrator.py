"""Basic orchestrator tests (no network)."""

import os

# Use isolated sqlite file for tests.
os.environ["DATABASE_URL"] = "sqlite:///./test_interview.db"
os.environ["SHARED_SECRET"] = "test-secret"

from app.db import SessionLocal, init_db
from app import orchestrator as orch
from app.auth import sign_payload
import time


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
        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "Big-O describes growth. Linear scan of an array is O(n) time and can be O(1) extra space if done in place.",
        )
        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "A stack is LIFO. Valid parentheses and DFS traversal are classic stack problems because order of nesting matters.",
        )
        assert view["stage"] in {"idea", "code", "qa"}
        # After enough Q&A we should be in coding path (or still probing if LLM online).
        if view["stage"] == "qa":
            row = orch.get_session(db, sid)
            view = orch.handle_message(
                db,
                row,
                "Recursion uses the call stack for frames; convert to iteration when depth is large or tail recursion applies.",
            )
        assert view["stage"] in {"idea", "code"}
        row = orch.get_session(db, sid)
        view = orch.handle_message(
            db,
            row,
            "I will use a hash map of value to index while scanning once. For each number check if target-num already seen. This is O(n) time and O(n) space. Edge cases: negatives and duplicates.",
        )
        # May still be idea if different problem; send another if needed.
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
        from datetime import datetime, timedelta, timezone

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
        assert view["duration_minutes"] == 17
        assert view["stage"] == "qa"
        spoken = " ".join(t["content"] for t in view["turns"] if t["role"] == "assistant")
        assert "nexai" in spoken.lower().replace(" ", "")
        assert view["ui"]["editor_locked"] is False

        sid = view["session_id"]
        row = orch.get_session(db, sid)
        row.created_at = datetime.now(timezone.utc) - timedelta(seconds=310)
        db.commit()
        db.refresh(row)
        view = orch.tick_session(db, row)
        assert view["stage"] == "idea"
        assert view["ui"]["show_editor"] is True
        assert view["ui"]["editor_locked"] is True
        assert "valid parentheses" in " ".join(t["content"] for t in view["turns"]).lower()

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

        row = orch.get_session(db, sid)
        row.created_at = datetime.now(timezone.utc) - timedelta(seconds=16 * 60)
        db.commit()
        db.refresh(row)
        view = orch.tick_session(db, row)
        assert view["status"] == "completed"
        last = view["turns"][-1]
        assert last["role"] == "assistant"
        assert last.get("meta", {}).get("spoken_wrap") or "nexai" in last["content"].lower()
    finally:
        db.close()


def test_sign():
    ts = int(time.time())
    sig = sign_payload(["start", 1, 2, 3, "sde_intern", 30, ts], secret="test-secret")
    assert len(sig) == 64
