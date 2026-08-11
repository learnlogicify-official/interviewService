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


def test_sign():
    ts = int(time.time())
    sig = sign_payload(["start", 1, 2, 3, "sde_intern", 30, ts], secret="test-secret")
    assert len(sig) == 64
