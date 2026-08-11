"""LLM client for dynamic interviewer turns."""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import get_settings

INTERVIEWER_SYSTEM = """You are a strict but fair technical interviewer for campus / early-career software roles.
Speak in short, natural spoken sentences (this is a voice interview).
Rules:
- NEVER give the solution, code, or step-by-step hints that solve the problem.
- NEVER repeat a question already asked in the transcript.
- Ask follow-ups based on what the candidate just said.
- If the answer is shallow, probe once; if still weak, move on.
- You may see the candidate's current editor code — ask about THEIR code only; do not rewrite it.
- Be professional and concise (2–5 sentences typically).

Always respond with a single JSON object only (no markdown fences):
{
  "reply": "what you say aloud to the candidate",
  "score": 0-100,
  "next_action": "followup" | "next_topic" | "move_to_coding" | "unlock_editor" | "probe_idea" | "continue_coding" | "wrap_up",
  "topic_tag": "short topic label"
}
"""


def llm_configured() -> bool:
    settings = get_settings()
    return bool(settings.openai_api_key and settings.openai_api_key.strip())


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def chat(messages: list[dict[str, str]], *, temperature: float = 0.55, max_tokens: int = 500) -> str | None:
    """Call chat completions; return assistant text or None on failure."""
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    base = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    try:
        with httpx.Client(timeout=35.0) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openai_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                },
            )
            # Some OpenAI-compatible providers reject response_format — retry without.
            if resp.status_code >= 400:
                resp = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.openai_model,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "messages": messages,
                    },
                )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def interviewer_turn(
    *,
    stage: str,
    role_track: str,
    topics: list[str],
    transcript: list[dict[str, str]],
    student_message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Ask the LLM for the next interviewer utterance + control signal.
    Returns None if LLM unavailable.
    """
    if not llm_configured():
        return None

    ctx = context or {}
    history_lines = []
    for t in transcript[-16:]:
        role = "Interviewer" if t.get("role") == "assistant" else "Candidate"
        history_lines.append(f"{role}: {t.get('content', '')}")
    history = "\n".join(history_lines) if history_lines else "(interview just started)"

    user_payload = {
        "stage": stage,
        "role_track": role_track,
        "topics": topics,
        "seconds_hint": ctx.get("seconds_remaining"),
        "qa_count": ctx.get("qa_index", 0),
        "problem": ctx.get("problem"),
        "current_code_excerpt": ctx.get("code_excerpt"),
        "idea_attempts": ctx.get("idea_attempts"),
        "transcript": history,
        "candidate_just_said": student_message,
        "stage_instructions": {
            "intro": "If they are ready, greet briefly and ask the first conceptual question. next_action=next_topic",
            "qa": "Evaluate their answer. Prefer a sharp follow-up (followup) OR a new related topic (next_topic). After about 3 solid exchanges or if time is low, next_action=move_to_coding.",
            "idea": "They must outline approach before coding. If solid, next_action=unlock_editor. If weak, next_action=probe_idea. Never reveal the optimal solution.",
            "code": "They are coding. Acknowledge briefly. Do not help. next_action=continue_coding unless they clearly want to finish (wrap_up).",
            "explain": "They explained a code excerpt. Score honesty/clarity. Then next_action=continue_coding.",
        }.get(stage, "Continue the interview professionally."),
    }

    raw = chat(
        [
            {"role": "system", "content": INTERVIEWER_SYSTEM},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.65,
        max_tokens=450,
    )
    data = _extract_json(raw or "")
    if not data or not data.get("reply"):
        return None

    action = str(data.get("next_action") or "followup").strip().lower()
    allowed = {
        "followup",
        "next_topic",
        "move_to_coding",
        "unlock_editor",
        "probe_idea",
        "continue_coding",
        "wrap_up",
    }
    if action not in allowed:
        action = "followup"

    try:
        score = float(data.get("score", 60))
    except Exception:
        score = 60.0
    score = max(0.0, min(100.0, score))

    return {
        "reply": str(data["reply"]).strip(),
        "score": score,
        "next_action": action,
        "topic_tag": str(data.get("topic_tag") or ""),
    }


def first_question(*, role_track: str, topics: list[str]) -> dict[str, Any] | None:
    """Generate the opening conceptual question dynamically."""
    if not llm_configured():
        return None
    raw = chat(
        [
            {"role": "system", "content": INTERVIEWER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "qa",
                        "role_track": role_track,
                        "topics": topics,
                        "candidate_just_said": "yes I am ready",
                        "stage_instructions": (
                            "Ask ONE strong opening conceptual interview question tailored to the role/topics. "
                            "next_action must be next_topic. Do not ask about coding problems yet."
                        ),
                        "transcript": "(start)",
                    }
                ),
            },
        ],
        temperature=0.7,
    )
    data = _extract_json(raw or "")
    if not data or not data.get("reply"):
        return None
    return {
        "reply": str(data["reply"]).strip(),
        "score": 0,
        "next_action": "next_topic",
        "topic_tag": str(data.get("topic_tag") or "opening"),
    }
