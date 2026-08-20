"""Gladia realtime speech-to-text (Live V2).

Dashboard / keys: https://app.gladia.io
API base: https://api.gladia.io
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("interview.gladia")


def gladia_configured() -> bool:
    return bool((get_settings().gladia_api_key or "").strip())


def _api_key() -> str:
    return (get_settings().gladia_api_key or "").strip().strip('"').strip("'")


def _base_url() -> str:
    settings = get_settings()
    return (settings.gladia_api_base or "https://api.gladia.io").rstrip("/")


def language_code(bcp47: str = "") -> str:
    """Map Moodle voicelang (e.g. en-IN) to Gladia ISO language code."""
    raw = (bcp47 or "en").strip().lower().replace("_", "-")
    if not raw:
        return "en"
    return raw.split("-", 1)[0] or "en"


def create_live_session(
    *,
    language: str = "en",
    sample_rate: int = 16000,
    session_id: str = "",
) -> dict[str, Any]:
    """
    POST /v2/live — returns a browser-safe WebSocket URL (token embedded).
    Never expose the Gladia API key to the client.
    """
    key = _api_key()
    if not key:
        return {"ok": False, "error": "GLADIA_API_KEY missing", "url": "", "id": ""}

    lang = language_code(language)
    payload: dict[str, Any] = {
        "model": "solaria-1",
        "encoding": "wav/pcm",
        "bit_depth": 16,
        "sample_rate": int(sample_rate or 16000),
        "channels": 1,
        # Wait for a real pause — 0.55s was cutting mid-thought.
        "endpointing": 1.4,
        # Safety only — if noise never looks like silence, don't hang forever.
        "maximum_duration_without_endpointing": 20,
        "language_config": {
            "languages": [lang],
            "code_switching": False,
        },
        "messages_config": {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            "receive_speech_events": True,
        },
        "realtime_processing": {
            "custom_vocabulary": True,
            "custom_vocabulary_config": {
                "vocabulary": [
                    "hashmap",
                    "Big-O",
                    "idempotency",
                    "React",
                    "JWT",
                    "RAG",
                    "NexPractice",
                    "NexAI",
                ],
            },
        },
    }
    if session_id:
        payload["custom_metadata"] = {"nexinterview_session": session_id[:64]}

    url = f"{_base_url()}/v2/live"
    headers = {
        "x-gladia-key": key,
        "Content-Type": "application/json",
    }

    def _post(body: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=body)
            if resp.status_code >= 400:
                err = f"HTTP {resp.status_code}: {resp.text[:400]}"
                logger.warning("Gladia live init failed: %s", err)
                return {"ok": False, "error": err, "url": "", "id": ""}
            data = resp.json() if resp.content else {}
            ws_url = str(data.get("url") or "").strip()
            job_id = str(data.get("id") or "").strip()
            if not ws_url:
                return {"ok": False, "error": "Gladia returned empty WebSocket URL", "url": "", "id": job_id}
            return {
                "ok": True,
                "id": job_id,
                "url": ws_url,
                "sample_rate": int(sample_rate or 16000),
                "language": lang,
                "provider": "gladia",
                "error": "",
            }

    try:
        result = _post(payload)
        if not result.get("ok") and "realtime_processing" in payload:
            # Free / strict plans may reject custom vocabulary — retry bare config.
            payload.pop("realtime_processing", None)
            result = _post(payload)
        return result
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        logger.warning("Gladia live init exception: %s", err)
        return {"ok": False, "error": err, "url": "", "id": ""}
