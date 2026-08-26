"""OpenAI (or compatible) text-to-speech."""

from __future__ import annotations

import base64
import logging

import httpx

from app.config import get_settings
from app.llm import _api_key, _set_error

logger = logging.getLogger("interview.tts")


def synthesize(text: str) -> dict:
    """
    Return {"ok": bool, "content_type": str, "audio_base64": str, "error": str}.
    Uses OpenAI /v1/audio/speech when key is configured.
    """
    settings = get_settings()
    key = _api_key()
    clean = " ".join((text or "").split())
    if len(clean) > 900:
        clean = clean[:900].rsplit(" ", 1)[0] + "."
    if not clean:
        return {"ok": False, "content_type": "", "audio_base64": "", "error": "empty text"}
    if not key:
        return {"ok": False, "content_type": "", "audio_base64": "", "error": "OPENAI_API_KEY missing"}

    base = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    # TTS is OpenAI-specific; if using a chat-only proxy, still hit api.openai.com for audio
    # unless OPENAI_TTS_BASE_URL is set.
    tts_base = (settings.openai_tts_base_url or base).rstrip("/")
    url = f"{tts_base}/audio/speech"
    model = settings.openai_tts_model or "tts-1"
    voice = settings.openai_tts_voice or "alloy"

    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "voice": voice,
                    "input": clean,
                    "response_format": "mp3",
                    "speed": 1.0,
                },
            )
            if resp.status_code >= 400:
                err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                _set_error(err)
                return {"ok": False, "content_type": "", "audio_base64": "", "error": err}
            audio = resp.content
            if not audio:
                return {"ok": False, "content_type": "", "audio_base64": "", "error": "empty audio"}
            return {
                "ok": True,
                "content_type": "audio/mpeg",
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "error": "",
            }
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        _set_error(err)
        logger.warning("TTS failed: %s", err)
        return {"ok": False, "content_type": "", "audio_base64": "", "error": err}
