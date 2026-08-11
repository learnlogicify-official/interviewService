"""SQLAlchemy models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    moodle_user_id: Mapped[int] = mapped_column(Integer, index=True)
    moodle_cm_id: Mapped[int] = mapped_column(Integer, index=True, default=0)
    moodle_instance_id: Mapped[int] = mapped_column(Integer, index=True, default=0)
    student_name: Mapped[str] = mapped_column(String(255), default="")
    role_track: Mapped[str] = mapped_column(String(64), default="sde_intern")
    status: Mapped[str] = mapped_column(String(32), default="active")  # active|completed|expired
    stage: Mapped[str] = mapped_column(String(32), default="intro")
    duration_minutes: Mapped[int] = mapped_column(Integer, default=30)
    topics_json: Mapped[str] = mapped_column(Text, default="[]")
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    recommendation: Mapped[str] = mapped_column(String(32), default="")
    report_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TurnRow(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str] = mapped_column(String(32), default="")
    role: Mapped[str] = mapped_column(String(16), default="assistant")  # assistant|student|system
    content: Mapped[str] = mapped_column(Text, default="")
    meta_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CodingSnapshotRow(Base):
    __tablename__ = "coding_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    problem_id: Mapped[str] = mapped_column(String(64), default="")
    code: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(32), default="autosave")  # autosave|run|submit
    run_result_json: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
