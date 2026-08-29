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
    duration_minutes: int = 17
    topics: list[str] = Field(default_factory=lambda: ["arrays", "strings", "hashmap", "stacks"])
    resume_text: str = ""
    moodle_problem_id: int = 0
    moodle_problem_title: str = ""
    interviewer_name: str = "NexAI"
    interviewer_style: str = "friendly"
    interviewer_briefing: str = ""
    include_coding: bool = True
    moodle_interviewer_id: int = 0
    difficulty: str = "intermediate"
    pace: str = "standard"
    question_mix: str = "conceptual"
    followup_depth: str = "moderate"
    avoid_topics: str = ""
    qa_minutes: int = 0
    timestamp: int
    signature: str


class MessageRequest(BaseModel):
    session_id: str
    message: str
    duration_sec: float = 0.0
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
    moodle_problem_id: int = 0
    ui: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    skill_graph: dict[str, Any] = Field(default_factory=dict)
    qa_topic: str = ""
    realtime_cue: str = ""
    awaiting_end_confirm: bool = False
    coding_just_passed: bool = False
    explain_excerpt: str = ""
    voice_metrics: dict[str, Any] = Field(default_factory=dict)
    turns: list[TurnOut] = Field(default_factory=list)
    report: dict[str, Any] | None = None


class LogTurnRequest(BaseModel):
    session_id: str
    content: str
    stage: str = ""
    timestamp: int
    signature: str


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


class SttRequest(BaseModel):
    session_id: str = ""
    audio_base64: str
    filename: str = "audio.webm"
    language: str = ""
    timestamp: int
    signature: str


class RealtimeTokenRequest(BaseModel):
    session_id: str
    timestamp: int
    signature: str


class GladiaLiveRequest(BaseModel):
    session_id: str
    language: str = "en"
    sample_rate: int = 16000
    timestamp: int
    signature: str


class CodingResultRequest(BaseModel):
    session_id: str
    passed: int = 0
    total: int = 0
    all_passed: bool = False
    problem_id: int = 0
    timestamp: int
    signature: str


class AssignProblemRequest(BaseModel):
    session_id: str
    problem_id: int
    problem_title: str = ""
    timestamp: int
    signature: str


class VoiceMetricsIn(BaseModel):
    """Optional client-side timing for communication metrics."""

    duration_sec: float = 0.0
