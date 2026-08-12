"""LLM client for dynamic interviewer turns."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("interview.llm")

# Last failure detail for /v1/health diagnostics (no secrets).
_LAST_ERROR: str = ""

INTERVIEWER_SYSTEM = """You are a strict but fair technical interviewer for campus / early-career software roles.
Speak like a real interviewer on a live voice call: short, natural, conversational sentences.
No markdown, no bullet lists, no stage directions, no "as an AI".
Rules:
- Ask EXACTLY ONE question per reply. Never stack a second question in the same reply.
- WAIT for the candidate's answer before moving on. Do not invent that they answered.
- If the candidate's message is empty, filler ("um", "ok", "yes" alone mid-QA), or nonsense, set next_action=followup, score under 30, and ask them to answer the SAME question again.
- NEVER give the solution, code, or step-by-step hints that solve the problem.
- NEVER repeat a question already asked in the transcript.
- NEVER reuse the same wording as a previous interviewer line.
- Ground every follow-up in the candidate's LAST answer: quote or paraphrase one concrete claim, then challenge correctness, edge cases, complexity, or trade-offs.
- Prefer concrete technical depth over generic praise. Avoid vague lines like "tell me more" without naming what to deepen.
- If the answer is shallow, probe once (followup); only then move to a NEW topic (next_topic).
- Only use move_to_coding after several real answered exchanges (not after noise).
- You may see the candidate resume — personalize lightly (projects/stack) but NEVER solve problems from it.
- You may see the candidate's current editor code — ask about THEIR code only; do not rewrite it.
- Keep replies to 1–3 short spoken sentences.

Always respond with a single JSON object only (no markdown fences):
{
  "reply": "what you say aloud to the candidate",
  "score": 0-100,
  "next_action": "followup" | "next_topic" | "move_to_coding" | "unlock_editor" | "probe_idea" | "continue_coding" | "wrap_up",
  "topic_tag": "short topic label"
}
"""

FILLER_RE = re.compile(
    r"^(um+|uh+|ah+|ok|okay|yes|yeah|yep|no|hmm+|mhm+|huh|what|repeat|again)[.!\s]*$",
    re.I,
)


def is_weak_answer(text: str, *, min_words: int = 6) -> bool:
    """True if the utterance should not advance the interview."""
    clean = " ".join((text or "").split()).strip()
    if len(clean) < 12:
        return True
    if FILLER_RE.match(clean):
        return True
    words = re.findall(r"[A-Za-z0-9_]+", clean)
    return len(words) < min_words


def _api_key() -> str:
    settings = get_settings()
    return (settings.openai_api_key or "").strip().strip('"').strip("'")


def llm_configured() -> bool:
    return bool(_api_key())


def last_error() -> str:
    return _LAST_ERROR


def _set_error(msg: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = (msg or "")[:500]
    if msg:
        logger.warning("LLM error: %s", _LAST_ERROR)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    # Strip accidental markdown fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
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
    key = _api_key()
    if not key:
        _set_error("OPENAI_API_KEY is empty")
        return None

    base = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload_base = {
        "model": settings.openai_model or "gpt-4o-mini",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json={**payload_base, "response_format": {"type": "json_object"}})
            if resp.status_code >= 400:
                # Retry without response_format (some proxies / older models).
                body1 = resp.text[:300]
                resp = client.post(url, headers=headers, json=payload_base)
                if resp.status_code >= 400:
                    _set_error(f"HTTP {resp.status_code} from {url}: {resp.text[:300] or body1}")
                    return None
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            content = str(content).strip()
            if not content:
                _set_error("LLM returned empty content")
                return None
            _set_error("")  # clear on success
            return content
    except Exception as exc:
        _set_error(f"{type(exc).__name__}: {exc}")
        return None


def ping() -> dict[str, Any]:
    """Tiny live call to verify Railway → OpenAI connectivity."""
    raw = chat(
        [
            {"role": "system", "content": 'Reply with JSON only: {"reply":"pong","score":100,"next_action":"followup","topic_tag":"ping"}'},
            {"role": "user", "content": "ping"},
        ],
        temperature=0,
        max_tokens=80,
    )
    data = _extract_json(raw or "")
    ok = bool(data and data.get("reply"))
    return {
        "ok": ok,
        "model": get_settings().openai_model,
        "base_url": (get_settings().openai_base_url or "").rstrip("/"),
        "raw_preview": (raw or "")[:160],
        "error": last_error() if not ok else "",
    }


def interviewer_turn(
    *,
    stage: str,
    role_track: str,
    topics: list[str],
    transcript: list[dict[str, str]],
    student_message: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Ask the LLM for the next interviewer utterance + control signal."""
    if not llm_configured():
        _set_error("OPENAI_API_KEY not configured")
        return None

    ctx = context or {}
    history_lines = []
    for t in transcript[-16:]:
        role = "Interviewer" if t.get("role") == "assistant" else "Candidate"
        history_lines.append(f"{role}: {t.get('content', '')}")
    history = "\n".join(history_lines) if history_lines else "(interview just started)"

    asked = ctx.get("asked_topics") or []
    resume = (ctx.get("resume_text") or "")[:4000]
    weak = is_weak_answer(student_message)
    user_payload = {
        "stage": stage,
        "role_track": role_track,
        "topics": topics,
        "already_covered_topics": asked,
        "seconds_hint": ctx.get("seconds_remaining"),
        "qa_count": ctx.get("qa_index", 0),
        "problem": ctx.get("problem"),
        "moodle_problem_id": ctx.get("moodle_problem_id"),
        "current_code_excerpt": ctx.get("code_excerpt"),
        "idea_attempts": ctx.get("idea_attempts"),
        "resume_excerpt": resume or None,
        "candidate_answer_looks_weak": weak,
        "skill_graph_summary": ctx.get("skill_graph_summary"),
        "claims": ctx.get("claims"),
        "voice_metrics_latest": ctx.get("voice_metrics_latest"),
        "transcript": history,
        "candidate_just_said": student_message,
        "stage_instructions": {
            "intro": "If they are ready, greet in ONE short spoken sentence then ask the first conceptual question. Sound natural, not like a script. next_action=next_topic",
            "qa": (
                "ONE question only. Sound like a human interviewer on a call — short spoken sentences, no markdown, no bullet lists. "
                "If candidate_answer_looks_weak is true, next_action MUST be followup "
                "and re-ask/clarify the same topic — do not move_to_coding. "
                "Otherwise evaluate: followup OR next_topic. Prefer probing WEAKEST skills from skill_graph_summary. "
                "If claims contain untested technology claims, drill one level deeper before moving on. "
                "Only after about 3 solid answered exchanges (qa_count), next_action=move_to_coding."
            ),
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
        temperature=0.7,
        max_tokens=450,
    )
    data = _extract_json(raw or "")
    if not data or not str(data.get("reply") or "").strip():
        if raw and not data:
            _set_error(f"Could not parse LLM JSON: {(raw or '')[:200]}")
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


def first_question(*, role_track: str, topics: list[str], resume_text: str = "") -> dict[str, Any] | None:
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
                        "resume_excerpt": (resume_text or "")[:4000] or None,
                        "candidate_just_said": "yes I am ready",
                        "stage_instructions": (
                            "Greet the candidate by first name in one short clause, then ask ONE strong "
                            "opening conceptual interview question tailored to the role/topics. "
                            "Spoken voice only: no markdown. Vary the topic (not always hash maps). "
                            "next_action must be next_topic. Do not ask about coding problems yet."
                        ),
                        "transcript": "(start)",
                    }
                ),
            },
        ],
        temperature=0.85,
        max_tokens=350,
    )
    data = _extract_json(raw or "")
    if not data or not str(data.get("reply") or "").strip():
        if raw and not data:
            _set_error(f"Could not parse opening JSON: {(raw or '')[:200]}")
        return None
    return {
        "reply": str(data["reply"]).strip(),
        "score": 0,
        "next_action": "next_topic",
        "topic_tag": str(data.get("topic_tag") or "opening"),
    }
