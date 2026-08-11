"""OpenAI Whisper speech-to-text."""

from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path

import httpx

from app.config import get_settings
from app.llm import _api_key, _set_error

logger = logging.getLogger("interview.stt")


def transcribe(audio_base64: str, *, filename: str = "audio.webm", language: str = "") -> dict:
    """
    Return {"ok": bool, "text": str, "error": str}.
    Expects base64-encoded audio (webm/ogg/wav/mp3).
    """
    settings = get_settings()
    key = _api_key()
    if not key:
        return {"ok": False, "text": "", "error": "OPENAI_API_KEY missing"}
    raw = (audio_base64 or "").strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        audio = base64.b64decode(raw, validate=False)
    except Exception as exc:
        return {"ok": False, "text": "", "error": f"bad_base64: {exc}"}
    if len(audio) < 64:
        return {"ok": False, "text": "", "error": "audio too short"}

    base = (settings.openai_stt_base_url or settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/audio/transcriptions"
    model = settings.openai_stt_model or "whisper-1"
    suffix = ".webm"
    lower = (filename or "").lower()
    for ext in (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".mp4"):
        if lower.endswith(ext):
            suffix = ext
            break

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = Path(tmp.name)
        try:
            with httpx.Client(timeout=90.0) as client:
                data = {"model": model}
                if language:
                    data["language"] = language.split("-")[0]
                files = {"file": (f"speech{suffix}", tmp_path.read_bytes(), "application/octet-stream")}
                resp = client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    data=data,
                    files=files,
                )
                if resp.status_code >= 400:
                    err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    _set_error(err)
                    return {"ok": False, "text": "", "error": err}
                payload = resp.json() if resp.content else {}
                text = str(payload.get("text") or "").strip()
                return {"ok": bool(text), "text": text, "error": "" if text else "empty transcript"}
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _set_error(err)
        logger.warning("STT failed: %s", err)
        return {"ok": False, "text": "", "error": err}
