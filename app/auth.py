"""HMAC auth between Moodle and interview-service."""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import HTTPException, status

from app.config import get_settings


def sign_payload(parts: list[str | int], secret: str | None = None) -> str:
    settings = get_settings()
    key = (secret or settings.shared_secret).encode("utf-8")
    msg = "|".join(str(p) for p in parts).encode("utf-8")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_signature(parts: list[str | int], signature: str, timestamp: int) -> None:
    settings = get_settings()
    now = int(time.time())
    if abs(now - int(timestamp)) > settings.token_ttl_seconds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    expected = sign_payload([*parts, timestamp])
    if not hmac.compare_digest(expected, signature or ""):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")
