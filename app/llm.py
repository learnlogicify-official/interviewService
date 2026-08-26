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

INTERVIEWER_SYSTEM = """You are a live voice technical interviewer for campus / early-career software roles.
Introduce yourself using the session interviewer name when provided. Sound human: vary greetings, never reuse a canned script.
Speak in 1–2 short sentences for acknowledgements. For code-snippet / predict-output turns you may use a short multi-line code block, then ONE clear question.
Voice pacing (critical): spoken words outside any code fence must stay under ~40 words; prefer ≤25 words for follow-ups.
One short acknowledgement then ONE question — never monologue, lecture, or stack multiple questions.
Never scold, shame, or lecture about "fuller answers" — stay calm and specific.
No markdown headings, no bullet lists, no "as an AI".
Shape (engine times this; you follow the stage):
- Conceptual / applied Q&A first. When this round closes, do not ask another technical question.
- If include_coding is true: later one coding problem; editor stays locked until approach is solid.
- If include_coding is false: stay on spoken / snippet Q&A until wrap.
- After unlock: ask about THEIR written code only.
Rules:
- Ask EXACTLY ONE question per reply, and wait for their answer.
- If they want to end early, ask them to confirm. Do not wrap_up unless they confirmed.
- NEVER give the solution, code, or step-by-step teaching hints that solve it.
- Probe, do not reveal. Prefer follow-ups that test understanding (DeepProbe style).
- Hint ladder (you may only operate H0–H3): H0 none, H1 clarify, H2 soft probe, H3 directed nudge. Never H4 near-solution.
- When question_node is provided AND dynamic_question_ok is false: stay on that competency; use spoken_now or a tight paraphrase.
- When dynamic_question_ok is true OR a faculty briefing is present: invent fresh questions that FOLLOW THE FACULTY BRIEFING first
  (topics, emphasis, difficulty, what to avoid). Do not ignore the briefing.
- Default question style is spoken conceptual / trade-off. Only use code-snippet, predict-output, or find-the-bug
  when suggested_format is predict|debug|complexity OR the briefing explicitly asks for that style.
- Do not jump to a NEW topic while the current question is still unanswered.
- EXCEPTION — unclear / incomplete / off-topic answers (critical for voice interviews):
  If the candidate's reply is garbled, half-finished, off-topic, or not related to what you asked:
  (1) briefly say you didn't catch a clear answer or it doesn't address the question,
  (2) REPHRASE the SAME question in simpler words (same competency / same resume claim),
  (3) set next_action=followup and keep score low (usually ≤35).
  Rephrasing the same question is REQUIRED here — it is not "repeating" in the bad sense.
- Only invent a brand-new topic when the previous answer was understandable AND on-topic
  (check transcript / already_covered_topics).
- NEVER paste or re-read a coding problem statement; it is already on their screen. Refer to it only by title.
- Ask EXACTLY ONE new question. Do not restack a previous stem plus a new one.
- Prefer probing WEAKEST skills from skill_graph_summary when choosing next_topic (engine may still override).
- Only set next_action=unlock_editor when the approach is actually clear enough to code.
- Do not set move_to_coding yourself; the engine closes the technical round.
- Score against rubric_strong / rubric_weak when present. Store topic_tag as the skill key when known.
- If the candidate says they don't know / are not aware / cannot answer / skips, score MUST be exactly 0.
  Do not award partial credit for admitting ignorance. next_action=followup or next_topic is fine,
  but the score stays near zero.
- Faculty briefing is the highest-priority topic guide when present — follow it closely.

Always respond with a single JSON object only (no markdown fences):
{
  "reply": "what you say aloud to the candidate",
  "score": 0-100,
  "next_action": "followup" | "next_topic" | "move_to_coding" | "unlock_editor" | "probe_idea" | "continue_coding" | "wrap_up",
  "topic_tag": "short topic label",
  "hint_level": 0
}
"""

FILLER_RE = re.compile(
    r"^(um+|uh+|ah+|ok|okay|yes|yeah|yep|no|hmm+|mhm+|huh|what|repeat|again)[.!\s]*$",
    re.I,
)

# Admissions of ignorance / no technical substance (still long enough to pass weak-gate).
NO_KNOWLEDGE_RE = re.compile(
    r"(?i)\b("
    r"i\s*(?:do\s*not|don't|dont)\s+know|"
    r"i\s*(?:am\s+)?not\s+(?:pretty\s+)?(?:much\s+)?aware|"
    r"not\s+aware|"
    r"completely\s+not\s+aware|"
    r"no\s+idea|"
    r"not\s+sure|"
    r"i\s*(?:have\s+)?no\s+(?:idea|clue|knowledge)|"
    r"i\s*(?:am|'m)\s+not\s+(?:that\s+)?strong|"
    r"not\s+that\s+strong|"
    r"i\s*(?:will\s+)?not\s+be\s+able\s+to\s+answer|"
    r"cannot\s+answer|can't\s+answer|"
    r"skip\s+(?:this|it)|"
    r"pass\s+on\s+this|"
    r"never\s+(?:studied|learned|heard)|"
    r"don'?t\s+remember|"
    r"no\s+experience\s+(?:with|in|on)"
    r")\b"
)


def is_weak_answer(text: str, *, min_words: int = 3) -> bool:
    """True only for filler / empty utterances — not truncated-but-real answers."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True
    if FILLER_RE.match(clean):
        return True
    words = re.findall(r"[A-Za-z0-9_]+", clean)
    if not words:
        return True
    # Keep this tight so a cut STT fragment still reaches the LLM instead of
    # the canned "I need a fuller answer" loop.
    if len(words) < min_words and len(clean) < 18:
        return True
    return False


def looks_incomplete_answer(text: str) -> bool:
    """True when speech looks cut off mid-thought (common with live STT)."""
    clean = " ".join((text or "").split()).strip()
    if not clean or is_weak_answer(clean):
        return False
    words = re.findall(r"[A-Za-z0-9_]+", clean)
    if len(words) <= 10 and not re.search(r"[.!?]$", clean):
        return True
    if re.search(
        r"(?i)\b(and|or|so|because|like|with|for|to|the|a|an|my|our|if|when|which)\s*$",
        clean,
    ):
        return True
    if clean.endswith(("...", "…", "-", "—")):
        return True
    return False


def is_no_knowledge_answer(text: str) -> bool:
    """
    True when the candidate explicitly declines / admits they cannot answer.
    These must score exactly 0 — no partial credit.
    """
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return True
    if NO_KNOWLEDGE_RE.search(clean):
        return True
    # Short shrugs that look like refusal without keywords.
    words = re.findall(r"[A-Za-z0-9_]+", clean.lower())
    if len(words) <= 12 and any(w in {"idk", "dunno"} for w in words):
        return True
    return False


def clamp_answer_score(text: str, score: float) -> float:
    """Cap scores so 'I don't know' cannot become 40–70."""
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if is_no_knowledge_answer(text):
        return 0.0
    if is_weak_answer(text):
        return min(s, 20.0)
    return max(0.0, min(100.0, s))


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
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    # Truncated JSON still often has a usable reply string.
    reply_m = re.search(r'"reply"\s*:\s*"((?:\\.|[^"\\])*)', text)
    if reply_m:
        try:
            reply = json.loads('"' + reply_m.group(1) + '"')
        except Exception:
            reply = reply_m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if str(reply).strip():
            return {
                "reply": str(reply).strip(),
                "score": 60,
                "next_action": "next_topic",
                "topic_tag": "",
                "hint_level": 0,
            }
    return None


def _compact_dossier(dossier: dict[str, Any] | None) -> dict[str, Any] | None:
    if not dossier or not isinstance(dossier, dict):
        return None
    projects = []
    for p in (dossier.get("projects") or [])[:8]:
        if isinstance(p, dict) and (p.get("name") or p.get("claim")):
            projects.append({
                "name": str(p.get("name") or "")[:80],
                "stack": [str(x)[:40] for x in (p.get("stack") or [])[:6]],
            })
    internships = []
    for p in (dossier.get("internships") or [])[:6]:
        if isinstance(p, dict):
            internships.append({
                "company": str(p.get("company") or "")[:80],
                "role": str(p.get("role") or "")[:80],
            })
    skills = []
    for s in (dossier.get("skills") or [])[:12]:
        if isinstance(s, dict) and s.get("name"):
            skills.append(str(s.get("name"))[:40])
        elif isinstance(s, str) and s.strip():
            skills.append(s.strip()[:40])
    plan = []
    for item in (dossier.get("question_plan") or [])[:8]:
        if isinstance(item, dict) and item.get("question"):
            plan.append({
                "anchor": str(item.get("anchor") or "")[:80],
                "question": str(item.get("question") or "")[:220],
            })
    return {
        "summary": str(dossier.get("summary") or "")[:280],
        "projects": projects,
        "internships": internships,
        "skills": skills,
        "question_plan": plan,
    }


def chat(messages: list[dict[str, str]], *, temperature: float = 0.55, max_tokens: int = 500, timeout: float = 18.0) -> str | None:
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
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json={**payload_base, "response_format": {"type": "json_object"}})
            if resp.status_code >= 400:
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
            _set_error("")
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


ANALYZE_SYSTEM = """You are a staff interviewer preparing a resume screen.
Read the resume carefully. Extract concrete facts only — never invent employers, projects, or tools.
Return JSON only with this shape:
{
  "summary": "one sentence about the candidate from the resume",
  "projects": [{"name": "", "stack": [""], "claim": "", "hard_questions": [""]}],
  "internships": [{"company": "", "role": "", "claim": "", "hard_questions": [""]}],
  "skills": [{"name": "", "evidence": "where it appears on the resume"}],
  "question_plan": [{"anchor": "exact project/company/skill from resume", "question": "one spoken interview question"}]
}
Rules for question_plan:
- 8 to 12 questions.
- Every question MUST name an anchor that appears in the resume text.
- Probe ownership, architecture, trade-offs, failure modes, metrics, and debugging.
- No generic DSA / hash-map / Big-O unless that topic is explicitly on the resume.
"""


def _is_human_resume_line(ln: str) -> bool:
    s = (ln or "").strip()
    if len(s) < 8:
        return False
    letters = len(re.findall(r"[A-Za-z]", s))
    if letters < 6:
        return False
    if letters / max(1, len(s)) < 0.42:
        return False
    if re.search(r"[ÿþßŠ¢ãÞµ«»¤¦§œžÐ]{2,}", s) and letters < 12:
        return False
    return True


def _heuristic_resume_dossier(resume_text: str) -> dict[str, Any]:
    lines = [re.sub(r"^[\-\*\u2022]+\s*", "", ln).strip() for ln in (resume_text or "").splitlines()]
    lines = [ln for ln in lines if _is_human_resume_line(ln)]
    projects: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []
    plan: list[dict[str, str]] = []
    skill_blob = re.search(
        r"(?is)(?:skills?|tech(?:nical)?\s+skills?|technologies)[:\n](.+?)(?:\n\n|\n[A-Z][A-Za-z ]{3,}:|$)",
        resume_text or "",
    )
    if skill_blob:
        for part in re.split(r"[,|/]| and ", skill_blob.group(1)):
            name = re.sub(r"\s+", " ", part).strip(" .;")
            if 2 <= len(name) <= 40:
                skills.append({"name": name, "evidence": "skills section"})
    for ln in lines:
        if re.search(
            r"(?i)\b(project|intern|developed|built|engineer|implemented|led|designed|application|platform)\b",
            ln,
        ):
            projects.append({
                "name": ln[:90],
                "stack": [],
                "claim": ln[:280],
                "hard_questions": [
                    f"On {ln[:60]}, what did you personally own versus the team, and what broke in production?"
                ],
            })
            if len(projects) >= 6:
                break
    for p in projects:
        plan.append({
            "anchor": str(p.get("name") or "this project"),
            "question": (p.get("hard_questions") or [""])[0],
        })
    for sk in skills[:6]:
        plan.append({
            "anchor": str(sk.get("name") or "this skill"),
            "question": (
                f"Your resume lists {sk.get('name')}. Give a concrete example from your work where you used it, "
                "including a trade-off you made."
            ),
        })
    if not plan and lines:
        plan.append({
            "anchor": lines[0][:80],
            "question": (
                f"On {lines[0][:70]}, walk me through the hardest technical decision you made, "
                "what you owned, and how you measured the result."
            ),
        })
    if not plan:
        plan.append({
            "anchor": "resume",
            "question": (
                "Walk me through the most technically demanding project on your resume — "
                "your role, the hardest bug, and how you measured success."
            ),
        })
    return {
        "summary": (lines[0][:180] if lines else "Resume on file."),
        "projects": projects,
        "internships": [],
        "skills": skills,
        "question_plan": plan[:12],
        "source": "heuristic",
    }


def _ensure_question_plan(dossier: dict[str, Any]) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    raw_plan = dossier.get("question_plan") or []
    for item in raw_plan:
        if isinstance(item, dict) and str(item.get("question") or "").strip():
            plan.append({
                "anchor": str(item.get("anchor") or "").strip()[:120],
                "question": str(item.get("question") or "").strip()[:400],
            })
        elif isinstance(item, str) and item.strip():
            plan.append({"anchor": "", "question": item.strip()[:400]})
    for bucket in ("projects", "internships"):
        for row in dossier.get(bucket) or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("company") or "").strip()
            for q in row.get("hard_questions") or []:
                if str(q).strip():
                    plan.append({"anchor": name[:120], "question": str(q).strip()[:400]})
    # Dedupe by question text.
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for item in plan:
        key = re.sub(r"\s+", " ", item["question"].lower())[:160]
        if key in seen:
            continue
        seen.add(key)
        if re.search(r"[ÿþßŠ¢ãÞµ«»¤¦§œžÐ]{2,}", item["question"]):
            continue
        if item.get("anchor") and not _is_human_resume_line(item["anchor"]) and len(item["anchor"]) > 12:
            item["anchor"] = ""
        out.append(item)
    return out[:12]


def analyze_resume(resume_text: str) -> dict[str, Any]:
    """Turn raw resume text into a structured interview dossier + question plan."""
    text = (resume_text or "").strip()
    fallback = _heuristic_resume_dossier(text)
    if len(text) < 40:
        return fallback
    if not llm_configured():
        return fallback
    try:
        raw = chat(
            [
                {"role": "system", "content": ANALYZE_SYSTEM},
                {"role": "user", "content": json.dumps({"resume": text[:6000]})},
            ],
            temperature=0.15,
            max_tokens=700,
            timeout=12.0,
        )
    except Exception as exc:
        _set_error(f"analyze_resume {type(exc).__name__}: {exc}")
        fallback["source"] = "heuristic_llm_failed"
        return fallback
    data = _extract_json(raw or "")
    if not data:
        fallback["source"] = "heuristic_llm_failed"
        return fallback
    dossier = {
        "summary": str(data.get("summary") or fallback.get("summary") or "")[:400],
        "projects": data.get("projects") if isinstance(data.get("projects"), list) else fallback["projects"],
        "internships": data.get("internships") if isinstance(data.get("internships"), list) else [],
        "skills": data.get("skills") if isinstance(data.get("skills"), list) else fallback["skills"],
        "question_plan": data.get("question_plan") if isinstance(data.get("question_plan"), list) else [],
        "source": "llm",
    }
    plan = _ensure_question_plan(dossier)
    if len(plan) < 3:
        plan = _ensure_question_plan(fallback) or plan
        dossier["source"] = "llm+heuristic"
    dossier["question_plan"] = plan
    return dossier


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
    for t in transcript[-8:]:
        role = "Interviewer" if t.get("role") == "assistant" else "Candidate"
        history_lines.append(f"{role}: {str(t.get('content', ''))[:360]}")
    history = "\n".join(history_lines) if history_lines else "(interview just started)"

    asked = ctx.get("asked_topics") or []
    resume = (ctx.get("resume_text") or "")[:8000]
    dossier = ctx.get("resume_dossier") or {}
    must_ask = ctx.get("must_ask_next") or {}
    weak = is_weak_answer(student_message)
    incomplete = looks_incomplete_answer(student_message)
    name = (ctx.get("interviewer_name") or "NexAI").strip() or "NexAI"
    style = (ctx.get("interviewer_style") or "friendly").strip().lower()
    briefing = (ctx.get("interviewer_briefing") or "").strip()
    include_coding = bool(ctx.get("include_coding", True))
    resume_only = bool(ctx.get("resume_only")) or str(role_track or "") == "resume_deep"
    dynamic_ok = bool(ctx.get("dynamic_question_ok", False)) or bool(briefing) or resume_only
    suggested_format = "concept" if resume_only else str(ctx.get("suggested_format") or "concept")

    style_line = {
        "friendly": "Tone: warm and encouraging, still rigorous.",
        "strict": "Tone: crisp and demanding. Push for precision; do not soft-pedal weak answers.",
        "brief": "Tone: concise. Keep acknowledgements short; questions can still include a tiny snippet.",
        "socratic": "Tone: Socratic. Ask why/how probes that make them reason; avoid lecturing.",
        "supportive": "Tone: calm and supportive. Rephrase gently; still score honestly.",
        "panel": "Tone: panel / hiring-bar. Ask for concrete evidence, trade-offs, and outcomes.",
    }.get(style, "Tone: professional and clear.")

    system = INTERVIEWER_SYSTEM
    extras: list[str] = []
    # Briefing first so the model treats it as primary.
    if briefing:
        extras.append(
            "FACULTY BRIEFING / CUSTOM INTERVIEWER RULES (MUST FOLLOW — this decides topics, "
            "difficulty, pace, mix, and what to avoid; do not substitute a generic DSA script):\n"
            f"{briefing[:2800]}"
        )
    difficulty = str(ctx.get("difficulty") or "intermediate")
    pace = str(ctx.get("pace") or "standard")
    question_mix = str(ctx.get("question_mix") or "conceptual")
    followup_depth = str(ctx.get("followup_depth") or "moderate")
    avoid_topics = str(ctx.get("avoid_topics") or "").strip()
    extras.extend(
        [
            f"Your spoken name for this session is {name}. Introduce yourself as {name}.",
            style_line,
            f"difficulty={difficulty}; pace={pace}; question_mix={question_mix}; followup_depth={followup_depth}",
            f"dynamic_question_ok={str(dynamic_ok).lower()}",
            f"suggested_format={suggested_format} "
            "(only use predict/debug/complexity snippet styles when this is one of those "
            "OR the briefing explicitly asks; otherwise ask conceptual/trade-off questions).",
        ]
    )
    if avoid_topics:
        extras.append(f"Do NOT ask about: {avoid_topics[:400]}")
    if not include_coding:
        extras.append(
            "This profile has coding DISABLED. Do not mention an editor, NexPractice, or unlocking code."
        )
    if resume_only:
        extras.append(
            "RESUME DEEP-DIVE MODE: You already analyzed the resume. Ask from resume_dossier / must_ask_next. "
            "Name the project, company, or skill in the question. Do not ask generic DSA unless it is on the resume. "
            "Do not use code-snippet predict-output formats."
        )
        summary = str((dossier or {}).get("summary") or "").strip()
        if summary:
            extras.append("Resume analysis summary: " + summary[:400])
        if must_ask and must_ask.get("question"):
            extras.append(
                "MUST ASK NEXT (paraphrase OK, keep the named anchor): "
                f"anchor={must_ask.get('anchor') or ''} | {must_ask.get('question')}"
            )
    if dynamic_ok:
        extras.append(
            "Invent questions that match the faculty briefing, custom rules, and topics list. "
            "Vary question angles every turn — do not reuse stems from the transcript. "
            "Do not fall back to generic hash-map / Big-O openers unless the briefing asks for them."
        )
    system = system + "\n\nSession profile:\n- " + "\n- ".join(extras)

    format_hint = {
        "concept": "Ask a conceptual / definition + example question aligned to the briefing.",
        "tradeoff": "Ask a trade-off or 'when would you choose X vs Y' question aligned to the briefing.",
        "predict": (
            "ONLY because suggested_format=predict: include a SHORT fenced code block (```), "
            "then ask them to predict the output. Do not reveal the answer."
        ),
        "debug": (
            "ONLY because suggested_format=debug: include a SHORT fenced buggy snippet (```), "
            "ask what is wrong. Do not reveal the fix."
        ),
        "complexity": (
            "ONLY because suggested_format=complexity: include a SHORT fenced snippet (```), "
            "ask time/space complexity."
        ),
    }.get(suggested_format, "Ask one solid technical question aligned to the briefing.")

    qa_instructions = (
        "Follow the faculty briefing for WHAT to ask. "
        f"This turn's format hint: {format_hint} "
        "If dynamic_question_ok is true, invent a NEW question matching briefing+topics "
        "(do not reuse stems from the transcript) — UNLESS the answer is weak/incomplete/off-topic. "
        "If candidate_answer_looks_weak OR candidate_answer_looks_incomplete is true, "
        "OR the answer does not address the last interviewer question: "
        "next_action MUST be followup — briefly say you didn't get a clear/on-topic answer, "
        "then REPHRASE the SAME question more simply (same topic). "
        "(bump hint_level by at most +1, max 3). Score low. "
        "Otherwise next_action=next_topic. Do not move_to_coding."
    )

    user_payload = {
        "stage": stage,
        "role_track": role_track,
        "topics": topics,
        "already_covered_topics": asked[-8:],
        "seconds_hint": ctx.get("seconds_remaining"),
        "qa_count": ctx.get("qa_index", 0),
        "problem": ctx.get("problem"),
        "moodle_problem_id": ctx.get("moodle_problem_id"),
        "current_code_excerpt": ctx.get("code_excerpt"),
        "idea_attempts": ctx.get("idea_attempts"),
        "resume_excerpt": resume[:2500] or None,
        "resume_dossier": _compact_dossier(dossier if isinstance(dossier, dict) else None),
        "must_ask_next": must_ask or None,
        "candidate_answer_looks_weak": weak,
        "candidate_answer_looks_incomplete": incomplete,
        "skill_graph_summary": ctx.get("skill_graph_summary"),
        "question_node": ctx.get("question_node"),
        "current_hint_level": ctx.get("current_hint_level", 0),
        "difficulty_ceiling": ctx.get("difficulty_ceiling"),
        "claims": (ctx.get("claims") or [])[:4],
        "interviewer_name": name,
        "interviewer_style": style,
        "include_coding": include_coding,
        "dynamic_question_ok": dynamic_ok,
        "suggested_format": suggested_format,
        "difficulty": difficulty,
        "pace": pace,
        "question_mix": question_mix,
        "followup_depth": followup_depth,
        "avoid_topics": avoid_topics or None,
        "transcript": history,
        "candidate_just_said": (student_message or "")[:800],
        "stage_instructions": {
            "intro": "Greet in ONE short spoken sentence then ask the first conceptual question. next_action=next_topic",
            "qa": (
                "RESUME MODE: next question must be grounded in resume_excerpt (a named project or skill). "
                "If candidate_answer_looks_weak OR candidate_answer_looks_incomplete, "
                "or the answer is off-topic: followup — say so briefly and rephrase the SAME resume question. "
                "Otherwise next_topic from another resume bullet. Do not move_to_coding."
                if resume_only
                else qa_instructions
            ),
            "idea": (
                "They must outline approach before coding. The full problem is already visible in the IDE — "
                "do NOT recite the problem statement. If solid, next_action=unlock_editor. "
                "If weak, next_action=probe_idea with H1–H3 nudge only. Never reveal the solution."
            ),
            "code": "They are coding. One short question about THEIR code, or a brief ack. next_action=continue_coding. If they asked to finish, next_action=wrap_up only to request confirmation.",
            "explain": "Score their explanation of the excerpt. Then next_action=continue_coding.",
        }.get(stage, "Continue the interview professionally."),
    }

    raw = chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        temperature=0.65 if dynamic_ok else 0.45,
        max_tokens=360 if resume_only else (280 if suggested_format in {"predict", "debug", "complexity"} else 220),
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

    try:
        hint_level = int(data.get("hint_level", ctx.get("current_hint_level", 0) or 0))
    except Exception:
        hint_level = int(ctx.get("current_hint_level", 0) or 0)
    hint_level = max(0, min(3, hint_level))  # model may not emit H4

    return {
        "reply": str(data["reply"]).strip(),
        "score": score,
        "next_action": action,
        "topic_tag": str(data.get("topic_tag") or ""),
        "hint_level": hint_level,
    }


def first_question(
    *,
    role_track: str,
    topics: list[str],
    resume_text: str = "",
    question_node: dict[str, Any] | None = None,
    interviewer_name: str = "NexAI",
    interviewer_style: str = "friendly",
    interviewer_briefing: str = "",
    include_coding: bool = True,
    suggested_format: str = "concept",
    resume_only: bool = False,
    resume_dossier: dict[str, Any] | None = None,
    must_ask_next: dict[str, Any] | None = None,
    dynamic_question_ok: bool = False,
    difficulty: str = "intermediate",
    pace: str = "standard",
    question_mix: str = "conceptual",
    followup_depth: str = "moderate",
    avoid_topics: str = "",
) -> dict[str, Any] | None:
    """Generate the opening conceptual question (anchored to question graph when provided)."""
    if not llm_configured():
        return None
    node = question_node or {}
    stem = str(node.get("spoken_now") or node.get("stem") or "").strip()
    name = (interviewer_name or "NexAI").strip() or "NexAI"
    briefing = (interviewer_briefing or "").strip()
    resume_only = bool(resume_only) or str(role_track or "") == "resume_deep"
    dynamic_ok = bool(dynamic_question_ok) or bool(briefing) or resume_only
    suggested_format = "concept" if resume_only else suggested_format
    coding_line = (
        "mention this is a resume deep-dive with no coding editor"
        if resume_only
        else (
            "mention a short technical screen then one coding problem"
            if include_coding
            else "mention this is a spoken technical screen without a coding editor"
        )
    )
    format_hint = {
        "predict": "ONLY if needed: open with a SHORT fenced code snippet and ask them to predict the output.",
        "debug": "ONLY if needed: open with a SHORT fenced buggy snippet and ask what is wrong.",
        "complexity": "ONLY if needed: open with a SHORT fenced snippet and ask time complexity.",
        "tradeoff": "Open with a trade-off question (X vs Y) matching the briefing.",
        "concept": "Open with a conceptual question matching the faculty briefing and topics.",
    }.get(suggested_format, "Open with a solid technical question matching the briefing.")

    system = INTERVIEWER_SYSTEM
    if briefing:
        system += (
            "\n\nFACULTY BRIEFING / CUSTOM INTERVIEWER RULES (MUST FOLLOW — primary guide):\n"
            f"{briefing[:2800]}"
        )
    system += (
        f"\n\nYour name is {name}. dynamic_question_ok={str(dynamic_ok).lower()}. "
        f"Style={interviewer_style}. Difficulty={difficulty}. Pace={pace}. "
        f"Question mix={question_mix}. Follow-up depth={followup_depth}. "
        f"suggested_format={suggested_format}."
    )
    if avoid_topics:
        system += f" Do NOT ask about: {avoid_topics[:400]}."
    if resume_only:
        system += (
            " RESUME MODE: You already analyzed their resume. The first question MUST name "
            "a project, internship, or skill from resume_dossier / must_ask_next."
        )

    if resume_only and must_ask_next and must_ask_next.get("question"):
        ask_rule = (
            "Ask this planned question (paraphrase OK, keep the named project/company/skill): "
            f"{must_ask_next.get('anchor') or ''} — {must_ask_next.get('question')}"
        )
    elif resume_only:
        ask_rule = (
            "Open by naming a specific project or internship from resume_dossier, then ask one rigorous "
            "question about THEIR contribution, a trade-off, or a failure they handled. "
            "Do not ask generic DSA."
        )
    elif dynamic_ok:
        ask_rule = (
            "Follow the faculty briefing and custom interviewer rules for topics and emphasis. "
            f"{format_hint} "
            "Do NOT default to generic hash-map / Big-O unless the briefing asks for it. "
            + (f"Optional bank skill hint: {stem}" if stem else "")
        )
    else:
        ask_rule = (
            f"You MUST ask this competency question (paraphrase OK): {stem}"
            if stem
            else "Ask one solid opening conceptual question for the role track."
        )

    raw = chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "qa",
                        "role_track": role_track,
                        "topics": topics,
                        "resume_excerpt": (resume_text or "")[:2500] or None,
                        "resume_dossier": _compact_dossier(resume_dossier),
                        "must_ask_next": must_ask_next or None,
                        "question_node": None if dynamic_ok else (node or None),
                        "dynamic_question_ok": dynamic_ok,
                        "suggested_format": suggested_format,
                        "include_coding": include_coding,
                        "resume_only": resume_only,
                        "difficulty": difficulty,
                        "pace": pace,
                        "question_mix": question_mix,
                        "followup_depth": followup_depth,
                        "candidate_just_said": "yes I am ready",
                        "stage_instructions": (
                            f"You are {name}. Give ONE clean spoken intro (greet once, say you are {name}, "
                            f"{coding_line}), then ask ONE technical question. "
                            "Do not stack two greetings. Do not add random filler. "
                            f"{ask_rule} "
                            "next_action=next_topic. hint_level=0."
                        ),
                        "transcript": "(start)",
                    }
                ),
            },
        ],
        temperature=0.55 if resume_only else (0.7 if dynamic_ok else 0.55),
        max_tokens=420,
        timeout=30.0,
    )
    data = _extract_json(raw or "")
    if not data or not str(data.get("reply") or "").strip():
        if raw and not data:
            _set_error(f"Could not parse opening JSON: {(raw or '')[:200]}")
        return None
    tag = str(data.get("topic_tag") or node.get("skill") or node.get("question_id") or "opening")
    reply = str(data["reply"]).strip()
    if resume_only and must_ask_next and must_ask_next.get("question"):
        planned = str(must_ask_next.get("question") or "").strip()
        if planned and "?" not in reply:
            reply = (reply.rstrip(" .") + ". " + planned).strip()
    return {
        "reply": reply,
        "score": 0,
        "next_action": "next_topic",
        "topic_tag": tag,
        "hint_level": 0,
        "question_id": "" if dynamic_ok else (node.get("question_id") or ""),
    }


def wrap_speech(
    *,
    student_name: str,
    scores: dict[str, Any],
    flags: list[str],
    evidence_tail: list[dict[str, Any]] | None = None,
    problem_titles: list[str] | None = None,
) -> str | None:
    """Short spoken closing feedback tailored to this session. Returns None if LLM unavailable."""
    if not llm_configured():
        return None
    first = (student_name or "there").split()[0]
    qa_n = int(scores.get("qa_answers") or 0)
    solved = int(scores.get("problems_solved") or 0)
    raw = chat(
        [
            {"role": "system", "content": INTERVIEWER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "wrap",
                        "stage_instructions": (
                            f"You are NexAI closing the interview with {first}. "
                            "Give a brief spoken recap in 3-5 sentences that is SPECIFIC to THESE scores "
                            "and evidence — never reuse a canned script. "
                            "If qa_answers is 0 and problems_solved is 0, say honestly that they didn't "
                            "engage enough to score and what to do next time. "
                            "Mention one strength only if a score is actually strong (>=70). "
                            "Mention the weakest 1-2 areas with concrete practice advice. "
                            "Do not ask another question. End by thanking them. next_action=wrap_up."
                        ),
                        "scores": scores,
                        "flags": flags,
                        "qa_answers": qa_n,
                        "problems_solved": solved,
                        "problem_titles": (problem_titles or [])[:4],
                        "evidence_tail": (evidence_tail or [])[-6:],
                    }
                ),
            },
        ],
        temperature=0.55,
        max_tokens=220,
    )
    data = _extract_json(raw or "")
    if data and str(data.get("reply") or "").strip():
        return str(data["reply"]).strip()
    if raw and not data:
        return " ".join(str(raw).split())[:600] or None
    return None
