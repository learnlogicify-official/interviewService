"""Session HTTP API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import verify_signature
from app.db import get_db
from app import orchestrator as orch
from app.schemas import (
    AssignProblemRequest,
    CodingResultRequest,
    EndSessionRequest,
    GladiaLiveRequest,
    MessageRequest,
    RealtimeTokenRequest,
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
    from app import __version__
    from app.config import get_settings
    from app.gladia import gladia_configured
    from app.llm import last_error, llm_configured

    settings = get_settings()
    base = (settings.openai_base_url or "").rstrip("/")
    return {
        "ok": True,
        "service": "interview-service",
        "version": __version__,
        "llm_configured": llm_configured(),
        "llm_model": settings.openai_model if llm_configured() else None,
        "llm_base_url": base if llm_configured() else None,
        "llm_last_error": last_error() or None,
        "tts_model": settings.openai_tts_model if llm_configured() else None,
        "stt_model": settings.openai_stt_model if llm_configured() else None,
        "stt_provider": settings.stt_provider,
        "gladia_configured": gladia_configured(),
        "realtime_model": settings.openai_realtime_model if llm_configured() else None,
        "voice_mode": settings.voice_mode,
    }


@router.get("/llm-ping")
def llm_ping() -> dict:
    """Live OpenAI connectivity check (no Moodle auth — use only while debugging)."""
    from app.llm import ping

    return ping()


@router.post("/realtime/token")
def realtime_token(body: RealtimeTokenRequest, db: Session = Depends(get_db)) -> dict:
    """Mint ephemeral OpenAI Realtime client secret for browser WebRTC."""
    from app import realtime as realtime_mod

    verify_signature(["realtime_token", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return realtime_mod.create_client_secret(
        session_id=row.id,
        student_name=row.student_name,
        role_track=row.role_track,
        stage=row.stage,
        moodle_user_id=row.moodle_user_id,
    )


@router.post("/gladia/live")
def gladia_live(body: GladiaLiveRequest, db: Session = Depends(get_db)) -> dict:
    """Mint Gladia Live V2 WebSocket URL for browser mic streaming (API key stays server-side)."""
    from app import gladia as gladia_mod

    verify_signature(["gladia_live", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return gladia_mod.create_live_session(
        language=body.language or "en",
        sample_rate=int(body.sample_rate or 16000),
        session_id=row.id,
    )


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
    try:
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
            moodle_problem_title=str(getattr(body, "moodle_problem_title", "") or ""),
            interviewer_name=str(getattr(body, "interviewer_name", "") or "NexAI"),
            interviewer_style=str(getattr(body, "interviewer_style", "") or "friendly"),
            interviewer_briefing=str(getattr(body, "interviewer_briefing", "") or ""),
            include_coding=bool(getattr(body, "include_coding", True)),
            moodle_interviewer_id=int(getattr(body, "moodle_interviewer_id", 0) or 0),
            difficulty=str(getattr(body, "difficulty", "") or "intermediate"),
            pace=str(getattr(body, "pace", "") or "standard"),
            question_mix=str(getattr(body, "question_mix", "") or "conceptual"),
            followup_depth=str(getattr(body, "followup_depth", "") or "moderate"),
            avoid_topics=str(getattr(body, "avoid_topics", "") or ""),
            qa_minutes=int(getattr(body, "qa_minutes", 0) or 0),
        )
        return SessionStateOut(**view)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"start_failed: {type(exc).__name__}: {exc}"[:400]) from exc


@router.post("/sessions/message", response_model=SessionStateOut)
def message(body: MessageRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(["message", body.session_id], body.signature, body.timestamp)
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(**orch.handle_message(db, row, body.message, duration_sec=float(body.duration_sec or 0)))


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


@router.post("/sessions/coding_result", response_model=SessionStateOut)
def coding_result(body: CodingResultRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(
        ["coding_result", body.session_id, int(body.passed), int(body.total)],
        body.signature,
        body.timestamp,
    )
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(
        **orch.handle_coding_result(
            db,
            row,
            passed=int(body.passed),
            total=int(body.total),
            all_passed=bool(body.all_passed),
            problem_id=int(body.problem_id or 0),
        )
    )


@router.post("/sessions/assign_problem", response_model=SessionStateOut)
def assign_problem(body: AssignProblemRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    verify_signature(
        ["assign_problem", body.session_id, int(body.problem_id)],
        body.signature,
        body.timestamp,
    )
    row = orch.get_session(db, body.session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionStateOut(
        **orch.assign_moodle_problem(
            db,
            row,
            problem_id=int(body.problem_id),
            problem_title=body.problem_title or "",
        )
    )


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
    return SessionStateOut(**orch.tick_session(db, row))


@router.get("/reports/cm/{cm_id}")
def reports(cm_id: int, timestamp: int, signature: str, db: Session = Depends(get_db)) -> dict:
    verify_signature(["reports", cm_id], signature, timestamp)
    return {"items": orch.list_reports_for_cm(db, cm_id)}


@router.get("/reports/user/{user_id}")
def reports_user(user_id: int, timestamp: int, signature: str, db: Session = Depends(get_db)) -> dict:
    verify_signature(["reports_user", user_id], signature, timestamp)
    return {"items": orch.list_reports_for_user(db, user_id)}
