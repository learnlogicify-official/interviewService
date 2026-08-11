"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    moodle_user_id: int
    moodle_cm_id: int = 0
    moodle_instance_id: int = 0
    student_name: str = "Student"
    role_track: str = "sde_intern"
    duration_minutes: int = 30
    topics: list[str] = Field(default_factory=lambda: ["arrays", "strings", "hashmap", "stacks"])
    timestamp: int
    signature: str


class MessageRequest(BaseModel):
    session_id: str
    message: str
    timestamp: int
    signature: str


class SnapshotRequest(BaseModel):
    session_id: str
    code: str
    source: str = "autosave"
    timestamp: int
    signature: str


class RunCodeRequest(BaseModel):
    session_id: str
    code: str
    mode: str = "sample"  # sample|hidden
    timestamp: int
    signature: str


class EndSessionRequest(BaseModel):
    session_id: str
    timestamp: int
    signature: str


class SessionViewRequest(BaseModel):
    session_id: str
    timestamp: int
    signature: str


class TurnOut(BaseModel):
    seq: int
    stage: str
    role: str
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionStateOut(BaseModel):
    session_id: str
    status: str
    stage: str
    duration_minutes: int
    seconds_remaining: int
    student_name: str
    role_track: str
    problem: dict[str, Any] | None = None
    ui: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    turns: list[TurnOut] = Field(default_factory=list)
    report: dict[str, Any] | None = None


class RunResultOut(BaseModel):
    ok: bool
    passed: int
    total: int
    details: list[dict[str, Any]]
    message: str = ""


class TtsRequest(BaseModel):
    session_id: str = ""
    text: str
    timestamp: int
    signature: str
