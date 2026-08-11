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
    SttRequest,
    TtsRequest,
)

router = APIRouter(prefix="/v1", tags=["sessions"])


@router.get("/health")
def health() -> dict:
    from app.config import get_settings
    from app.llm import last_error, llm_configured

    settings = get_settings()
    base = (settings.openai_base_url or "").rstrip("/")
    return {
        "ok": True,
        "service": "interview-service",
        "version": "0.3.1",
        "llm_configured": llm_configured(),
        "llm_model": settings.openai_model if llm_configured() else None,
        "llm_base_url": base if llm_configured() else None,
        "llm_last_error": last_error() or None,
        "tts_model": settings.openai_tts_model if llm_configured() else None,
        "stt_model": settings.openai_stt_model if llm_configured() else None,
    }


@router.get("/llm-ping")
def llm_ping() -> dict:
    """Live OpenAI connectivity check (no Moodle auth — use only while debugging)."""
    from app.llm import ping

    return ping()


@router.post("/tts")
def tts(body: TtsRequest) -> dict:
    """Return base64 MP3 for interviewer speech (OpenAI TTS)."""
    from app import tts as tts_mod

    verify_signature(["tts", body.session_id or "-", body.text[:80]], body.signature, body.timestamp)
    result = tts_mod.synthesize(body.text)
    return result


@router.post("/stt")
def stt(body: SttRequest) -> dict:
    """Transcribe candidate audio via OpenAI Whisper."""
    from app import stt as stt_mod

    verify_signature(
        ["stt", body.session_id or "-", len(body.audio_base64 or "")],
        body.signature,
        body.timestamp,
    )
    return stt_mod.transcribe(
        body.audio_base64,
        filename=body.filename or "audio.webm",
        language=body.language or "",
    )


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
        resume_text=body.resume_text or "",
        moodle_problem_id=int(body.moodle_problem_id or 0),
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


@router.get("/reports/user/{user_id}")
def reports_user(user_id: int, timestamp: int, signature: str, db: Session = Depends(get_db)) -> dict:
    verify_signature(["reports_user", user_id], signature, timestamp)
    return {"items": orch.list_reports_for_user(db, user_id)}
