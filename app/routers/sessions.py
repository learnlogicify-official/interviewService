"""Session HTTP API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import verify_signature
from app.db import get_db
from app import orchestrator as orch
from app.schemas import (
    EndSessionRequest,
    MessageRequest,
    RunCodeRequest,
    RunResultOut,
    SessionStateOut,
    SessionViewRequest,
    SnapshotRequest,
    StartSessionRequest,
)

router = APIRouter(prefix="/v1", tags=["sessions"])


@router.get("/health")
def health() -> dict:
    from app.llm import llm_configured

    return {
        "ok": True,
        "service": "interview-service",
        "llm_configured": llm_configured(),
    }


@router.post("/sessions/start", response_model=SessionStateOut)
def start(body: StartSessionRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(
        [
            "start",
            body.moodle_user_id,
            body.moodle_cm_id,
            body.moodle_instance_id,
            body.role_track,
            body.duration_minutes,
        ],
        body.signature,
        body.timestamp,
    )
    view = orch.start_session(
        db,
        moodle_user_id=body.moodle_user_id,
        moodle_cm_id=body.moodle_cm_id,
        moodle_instance_id=body.moodle_instance_id,
        student_name=body.student_name,
        role_track=body.role_track,
        duration_minutes=body.duration_minutes,
        topics=body.topics,
    )
    return SessionStateOut(**view)


@router.post("/sessions/message", response_model=SessionStateOut)
def message(body: MessageRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(["message", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(**orch.handle_message(db, row, body.message))


@router.post("/sessions/snapshot", response_model=SessionStateOut)
def snapshot(body: SnapshotRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(["snapshot", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(**orch.save_snapshot(db, row, body.code, body.source))


@router.post("/sessions/run")
def run(body: RunCodeRequest, db: Session = Depends(get_db)) -> dict:
    verify_signature(["run", body.session_id, body.mode], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    view, result = orch.run_code(db, row, body.code, body.mode)
    return {"session": SessionStateOut(**view).model_dump(), "result": RunResultOut(**result).model_dump()}


@router.post("/sessions/end", response_model=SessionStateOut)
def end(body: EndSessionRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(["end", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(**orch.finish_session(db, row, reason="student_ended"))


@router.post("/sessions/get", response_model=SessionStateOut)
def get_one(body: SessionViewRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(["get", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    if orch._expire_if_needed(db, row):
        db.refresh(row)
    return SessionStateOut(**orch.session_view(db, row))


@router.get("/reports/cm/{cm_id}")
def reports(cm_id: int, timestamp: int, signature: str, db: Session = Depends(get_db)) -> dict:
    verify_signature(["reports", cm_id], signature, timestamp)
    return {"items": orch.list_reports_for_cm(db, cm_id)}
