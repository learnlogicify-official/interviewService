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
Introduce yourself as NexAI only (never a custom profile title). Sound human: vary greetings, never reuse a canned script.
Speak in 1–2 short sentences for acknowledgements. For code-snippet / predict-output turns you may use a short multi-line code block, then ONE clear question.
Voice pacing: acknowledgements ≤12 words. Questions 12–28 spoken words — one situation, one ask. Follow-ups ≤20 words.
Talk like a sharp human on a video call, not a written exam. Contractions. Vary bridges.
Ask as if you just thought of the scenario ("Say you're…", "Quick one —", "Imagine…").
One short acknowledgement then ONE question — never monologue, lecture, or stack multiple questions.
LANGUAGE: English only for every spoken line, caption, and reply. Never Tamil, Hindi, Chinese, or any other language.
No markdown headings, no bullet lists, no "as an AI".

TONE (applies to EVERY turn, including weak answers and topic changes):
- Hold the configured tone for the whole session. A friendly interviewer stays friendly at minute 25,
  even after three bad answers. Never drift into blunt, clipped, or dismissive phrasing.
- Never scold, shame, grade aloud, or lecture ("fuller answers", "take this seriously",
  "we'll mark this topic as weak", "you weren't ready"). Scores are private — never say them.
- When you move on after a weak answer: one short neutral-positive bridge ("No problem, let's switch."),
  then the next question. Never announce that they failed the topic.
- If the candidate says your question was unclear or does not make sense: own it in three or four words
  ("Fair — let me make it concrete."), then ask a fully re-specified version. Never argue or repeat the
  same vague wording.

QUESTION QUALITY (a vague question is a failed turn):
- Every question must be CONCRETE and self-contained: name the topic AND anchor it to something specific —
  a small scenario, real inputs/numbers, a named pair of options, an error/behaviour to explain, or a
  decision they must justify. The candidate should never have to guess what you are asking.
- NEVER ask these shapes: "How would you use X in a project?", "How would you use X in a small project?",
  "Can you explain how X is implemented?", "Walk me through how you'd approach it", "Tell me about X",
  "What are the key operations of X?", "Explain the difference between X and Y" with no scenario,
  or any bare textbook-definition question. They are too generic for an interview.
- Do not ask for a definition of something the candidate already defined correctly. Move the angle
  forward: mechanism, failure mode, trade-off with a named alternative, or a concrete design choice.
- Do not re-ask a question already in the transcript or in already_asked_questions, and do not ask a
  thin variation of one. Each turn must open new ground.
- Stay answerable in 20–60 seconds of speech. No multi-part questions.
- A named topic alone is NOT a question. "Let's talk about Java" must be followed by a specific ask.
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
- When dynamic_question_ok is true OR a faculty briefing is present: invent fresh questions that FOLLOW THE FACULTY BRIEFING /
  CUSTOM INTERVIEWER RULES first (topics, emphasis, difficulty, what to avoid). Do not ignore them.
- Never default to generic hash-map / Big-O / sorted-array openers unless those topics are explicitly listed.
- Prefer: name a focus topic → ask for a concrete scenario, trade-off, edge case, or failure mode matching difficulty.
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
- Faculty briefing / custom interviewer rules are the highest-priority topic guide when present — follow them closely.

Calibration examples (shape only — always match the configured topics):
BAD: "How would you use Java in a small project? Any trade-offs?"
GOOD: "Say you remove() from an ArrayList while you're looping it — what blows up, and how do you actually delete those rows?"
BAD: "Can you explain how a queue is implemented in programming?"
GOOD: "Your array-backed queue holds 100 slots. After 100 enqueues and 60 dequeues, enqueue fails even though 60 slots are free. What's going on?"
BAD: "Explain the difference between a stack and a queue."
GOOD: "Build-system tasks should run oldest-first. Stack or queue — and what breaks if you pick the other one?"

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

PHANTOM_STT_RE = re.compile(
    r"(?i)("
    r"thank you for watching|thanks for watching|"
    r"like,?\s*comment\s*(and|&)?\s*subscribe|"
    r"don'?t forget to like|smash that like|"
    r"please subscribe|thanks for listening|"
    r"see you in the next"
    r")"
)


def is_phantom_transcript(text: str) -> bool:
    """True for YouTube-style STT hallucinations on silence."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return False
    return bool(PHANTOM_STT_RE.search(clean))

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


# The candidate is objecting to the QUESTION, not admitting they don't know the topic.
# These must never score 0 — the interviewer owes them a concrete re-specification.
UNCLEAR_QUESTION_RE = re.compile(
    r"(?i)("
    r"(?:did\s*n[o']?t|didn'?t|not)\s+(?:really\s+|quite\s+|fully\s+)?"
    r"(?:get|catch|follow|understand|understood)\s+(?:your|the|that|this)\s*(?:question|point|ask)|"
    r"i\s+(?:really\s+)?(?:didn'?t|did\s+not)\s+get\s+(?:your|the)?\s*question|"
    r"i\s+didn'?t\s+get\s+you|"
    r"didn'?t\s+get\s+you|"
    r"didn'?t\s+get\s+(?:that|this|it)\b|"
    r"what\s+is\s+meant\s+by|"
    r"(?:i\s+)?don'?t\s+(?:really\s+|quite\s+)?understand(?:\s+really)?(?:\s+(?:your|the|that|this|by))|"
    r"(?:i\s+)?don'?t\s+understand\s+really|"
    r"(?:your|that|the|this)\s+question\s+(?:itself\s+)?(?:is|was|sounds|seems|makes)"
    r"\s*(?:n['o]?t)?\s*(?:absurd|vague|unclear|confusing|weird|generic|no\s+sense|not\s+clear)|"
    r"question\s+itself\s+is\s+absurd|"
    r"(?:doesn'?t|does\s+not)\s+make\s+(?:any\s+)?sense|"
    r"what\s+do\s+you\s+mean\s+by|"
    r"what\s+this\s+meant|"
    r"can\s+you\s+(?:please\s+)?(?:be\s+more\s+specific|rephrase|clarify)|"
    r"(?:could|can)\s+you\s+repeat\s+(?:the|that)\s+question|"
    r"didn'?t\s+get\s+(?:your|the)\s+question"
    r")"
)

SKIP_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"(?:can we |let'?s |please )?(?:go to |move (?:on )?to |skip (?:to )?)(?:the )?next"
    r"(?: topic| one| question)?"
    r"|next\s+(?:topic|one|question|stop)"
    r"|skip this(?: one| question| topic)?"
    r"|move on"
    r")\b"
)

# Question shapes that are too generic to speak in a real interview.
# Match even without a '?' — the old offline template never used one.
VAGUE_QUESTION_RE = re.compile(
    r"(?i)("
    r"how\s+would\s+you\s+use\s+\w[\w\s]{0,40}\s+in\s+a\s+(?:small\s+|simple\s+|real\s+)?project|"
    r"can\s+you\s+explain\s+how\s+\w[\w\s]{0,40}\s+is\s+implemented|"
    r"how\s+(?:is|would\s+you\s+implement)\s+\w[\w\s]{0,40}"
    r"\s+(?:in\s+programming|using\s+an\s+array)|"
    r"walk\s+me\s+through\s+how\s+you'?d?\s+approach\s+it|"
    r"approach\s+it\s+in\s+a\s+real\s+project|"
    r"what\s+are\s+the\s+key\s+operations|"
    r"(?:explain|describe|tell\s+me)\s+(?:the\s+)?(?:difference|differences)\s+between|"
    r"tell\s+me\s+about\s+\w[\w\s]{0,24}$|"
    r"explain\s+big-?\s*o|"
    r"let'?s\s+start\s+with\s+.{0,90}walk\s+me\s+through|"
    r"can\s+you\s+describe\s+how\s+you\s+would\s+implement|"
    r"how\s+do\s+you\s+handle\s+overflow\s+in\s+this"
    r")"
)

RUDE_SPOKEN_RE = re.compile(
    r"(?i)("
    r"we(?:'ll| will) mark this topic as weak|"
    r"you weren'?t ready|"
    r"take this (?:more )?seriously|"
    r"need a fuller answer|"
    r"that(?:'s| is) a weak (?:answer|topic)|"
    r"i(?:'m| am) marking (?:this|that)|"
    r"noted\s*\(\s*\d|"
    r"got it\s*\(\s*\d"
    r")"
)

SCORE_SPOKEN_RE = re.compile(r"\(\s*\d+(?:\.\d+)?\s*/\s*100\s*\)")

DSA_DRIFT_RE = re.compile(
    r"(?i)\b("
    r"big-?\s*o|time complexity|space complexity|"
    r"hash\s*-?maps?|hashmap|"
    r"difference between (?:a )?stack and (?:a )?queue|"
    r"stack and (?:a )?queue|"
    r"enqueue|dequeue|"
    r"array-backed queue|array-based queue|"
    r"implement(?:ed)? (?:a |the )?queue|"
    r"o\(\s*n\b"
    r")\b"
)

_DSA_TOPIC_KEYS = (
    "dsa",
    "data structure",
    "algorithm",
    "complexity",
    "big-o",
    "big o",
    "hash",
    "stack",
    "queue",
    "linked list",
    "binary tree",
    "graph",
    "recursion",
    "dynamic programming",
)


def is_unclear_question_response(text: str) -> bool:
    """True when the candidate is saying the interviewer's question was unclear/absurd."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return False
    return bool(UNCLEAR_QUESTION_RE.search(clean))


def is_skip_topic_request(text: str) -> bool:
    """True when the candidate asks to leave this topic and go to the next one."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return False
    return bool(SKIP_TOPIC_RE.search(clean))


def is_vague_question(text: str, *, min_words: int = 0) -> bool:
    """
    True when a generated interviewer question is too generic to speak.

    min_words guards against thin stems; keep it 0 for follow-up probes, where a
    short "why does that happen?" is a legitimate deepening question.
    Banned shapes match even without a '?' — the old offline template never used one.
    """
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return False
    if VAGUE_QUESTION_RE.search(clean):
        return True
    if "?" not in clean:
        return False
    # Only judge the question sentence itself, not the acknowledgement before it.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if "?" in p]
    target = parts[-1] if parts else clean
    if VAGUE_QUESTION_RE.search(target):
        return True
    if min_words > 0:
        words = re.findall(r"[A-Za-z0-9_]+", target)
        return len(words) < min_words
    return False


def is_off_lock_dsa(text: str, allowed_topics: list[str] | None) -> bool:
    """True when the spoken question drifted into generic DSA off the configured list."""
    topics = [str(t).lower() for t in (allowed_topics or []) if str(t).strip()]
    if not topics:
        return False
    blob = " ".join(topics)
    if any(k in blob for k in _DSA_TOPIC_KEYS):
        return False
    return bool(DSA_DRIFT_RE.search(text or ""))


def strip_spoken_meta(text: str) -> str:
    """Drop rude grading lines, spoken scores, and leaked API errors. Never let those reach the mic."""
    clean = " ".join((text or "").split()).strip()
    if not clean:
        return ""
    # Never speak raw OpenAI / HTTP failure blobs mid-interview.
    clean = re.sub(
        r"(?is)\(\s*HTTP\s*\d{3}[^)]*\)",
        "",
        clean,
    )
    clean = re.sub(
        r"(?is)\bhttps?://\S*(?:openai|api)\S*",
        "",
        clean,
    )
    clean = re.sub(
        r"(?is)\{\s*\"error\"[\s\S]{0,400}",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)\byou have no credits remain\w*\b[^.!?]*[.!]?",
        "",
        clean,
    )
    clean = re.sub(
        r"(?i)\brate limit reached\b[^.!?]*[.!]?",
        "",
        clean,
    )
    clean = " ".join(clean.split()).strip()
    parts = re.split(r"(?<=[.!?])\s+", clean)
    kept: list[str] = []
    for part in parts:
        piece = SCORE_SPOKEN_RE.sub("", part).strip(" -—")
        if not piece or RUDE_SPOKEN_RE.search(piece):
            continue
        kept.append(piece)
    out = " ".join(kept).strip()
    # Mid-session turns must not re-introduce NexAI ("Hi, I'm NexAI…").
    out = re.sub(
        r"(?i)^(hi|hey|hello|welcome)[,!]?\s+(?:there[,!]?\s+)?i(?:'m| am)\s+nexai\b[^.!?]*[.!]?\s*",
        "",
        out,
    ).strip()
    if re.search(
        r"(?i)(\bon\s+\w[\w ]{0,40},\s+what did you personally own|\bused it\b.*\bnearly went wrong\b|"
        r"\bstaying on self[- ]introduction\b)",
        out,
    ):
        out = _repair_resume_question(out)
    return out


def default_ack(style: str = "friendly") -> str:
    """Short spoken acknowledgement that matches the configured interviewer tone."""
    key = (style or "friendly").strip().lower()
    if key in {"friendly", "supportive"}:
        return "Thanks —"
    if key == "socratic":
        return "Okay —"
    return "Got it."


def is_no_knowledge_answer(text: str) -> bool:
    """
    True when the candidate explicitly declines / admits they cannot answer.
    These must score exactly 0 — no partial credit.
    """
    if is_unclear_question_response(text):
        return False
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
                # One retry: rate-limit keeps JSON mode; other 4xx fall back once without it.
                if resp.status_code in {429, 502, 503}:
                    import time
                    time.sleep(1.2 if resp.status_code == 429 else 0.4)
                    resp = client.post(
                        url,
                        headers=headers,
                        json={**payload_base, "response_format": {"type": "json_object"}},
                    )
                elif resp.status_code in {400, 422}:
                    resp = client.post(url, headers=headers, json=payload_base)
                if resp.status_code >= 400:
                    _set_error(f"HTTP {resp.status_code} from LLM provider")
                    logger.warning("LLM HTTP %s: %s", resp.status_code, (resp.text[:300] or body1))
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


_LEADING_PROJECT_VERBS = re.compile(
    r"^(?:i\s+)?(?:have\s+)?(?:also\s+)?"
    r"(?:developed|built|designed|implemented|created|worked\s+on|led|engineered|"
    r"contributed\s+to|made|wrote|coding|project|application|platform)\s*[:\-]?\s*",
    re.I,
)
_STOP_TITLE_WORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with", "using",
    "from", "my", "our", "this", "that", "into", "over", "under", "via",
}


def _project_title_from_line(ln: str) -> str:
    """
    Pull a short project title from a resume bullet.

    Avoids spoken questions like "On Developed, …" or "On Driven Architecture Platform, …"
    when the line is a long sentence starting with a verb.
    """
    s = re.sub(r"^[\-\*\u2022]+\s*", "", (ln or "").strip())
    s = _LEADING_PROJECT_VERBS.sub("", s).strip(" :-–—")
    quoted = re.search(r"[\"'“]([^\"'”]{3,60})[\"'”]", s)
    if quoted:
        return quoted.group(1).strip()
    # "Title — description" / "Title: description"
    headed = re.match(
        r"^([A-Za-z][A-Za-z0-9+][\w ./\-+#]{1,50}?)(?:\s*[\-–—:|]\s+|\s+\(|$)",
        s,
    )
    if headed:
        title = headed.group(1).strip(" .")
        words = title.split()
        if 1 <= len(words) <= 8 and title.lower() not in {
            "developed", "built", "project", "application", "platform", "system",
        }:
            return title
    words = [
        w for w in re.findall(r"[A-Za-z][A-Za-z0-9+.#]*", s)
        if w.lower() not in _STOP_TITLE_WORDS
    ]
    if not words:
        return "your project"
    title = " ".join(words[:5])
    if title.lower() in {"developed", "built", "platform", "application", "system", "project"}:
        return "your project"
    return title


def _repair_resume_question(question: str, anchor: str = "") -> str:
    """Rewrite broken resume probes so candidates hear a clear, named ask."""
    q = " ".join((question or "").split()).strip()
    if not q:
        return q
    anchor_title = _project_title_from_line(anchor) if anchor else ""
    if anchor_title.lower() in {"", "your project"}:
        anchor_title = ""

    # "On Developed, what did you personally own…" / truncated mid-phrase titles
    m = re.match(r"(?i)^on\s+(.+?),\s*(what did you personally own.+)$", q)
    if m:
        raw_name, rest = m.group(1).strip(), m.group(2).strip()
        name = anchor_title or _project_title_from_line(raw_name)
        if name.lower() in {"developed", "built", "your project"} or len(name.split()) == 1 and len(name) < 6:
            return (
                "Pick one project from your resume. What did you personally own end to end, "
                "and what broke or nearly broke when real users or real data hit it?"
            )
        rest = rest[0].lower() + rest[1:] if rest else rest
        return f"For your {name} project, {rest}"

    # "…where you used it" with no skill/project referent
    if re.search(r"(?i)\bused it\b", q) and not re.search(
        r"(?i)\b(?:java|python|react|kafka|sql|node|spring|docker|aws|[\w+]{4,})\b.{0,40}\bused it\b",
        q,
    ):
        if anchor_title:
            return (
                f"Your resume mentions {anchor_title}. Give one concrete situation where you used it, "
                "the decision you made, and what nearly went wrong."
            )
        return (
            "Pick one skill or project from your resume. Give a concrete situation where you used it, "
            "the decision you made, and what nearly went wrong."
        )

    # "Staying on self-introduction: … used it" style mashups
    if re.search(r"(?i)\bself[- ]introduction\b", q) and re.search(r"(?i)\bused it\b", q):
        return (
            "Introduce yourself briefly, then describe one specific situation from your resume: "
            "the decision you had to make, and what nearly went wrong."
        )
    return q


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
            title = _project_title_from_line(ln)
            projects.append({
                "name": title[:90],
                "stack": [],
                "claim": ln[:280],
                "hard_questions": [
                    _repair_resume_question(
                        f"On {title[:60]}, what did you personally own versus the team, and what broke in production?",
                        title,
                    )
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
        sk_name = str(sk.get("name") or "").strip()
        if len(sk_name) < 2:
            continue
        plan.append({
            "anchor": sk_name,
            "question": (
                f"Your resume lists {sk_name}. Give a concrete example from your work where you used it, "
                "including a trade-off you made."
            ),
        })
    if not plan and lines:
        title = _project_title_from_line(lines[0])
        plan.append({
            "anchor": title[:80],
            "question": (
                f"On {title[:70]}, walk me through the hardest technical decision you made, "
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
    # Final pass: repair any leftover broken stems.
    cleaned_plan: list[dict[str, str]] = []
    for item in plan:
        q = _repair_resume_question(str(item.get("question") or ""), str(item.get("anchor") or ""))
        if q:
            cleaned_plan.append({"anchor": str(item.get("anchor") or "")[:120], "question": q[:400]})
    return {
        "summary": (lines[0][:180] if lines else "Resume on file."),
        "projects": projects,
        "internships": [],
        "skills": skills,
        "question_plan": cleaned_plan[:12],
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
        item["question"] = _repair_resume_question(
            str(item.get("question") or ""),
            str(item.get("anchor") or ""),
        )
        if not item["question"]:
            continue
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


OBSERVER_SYSTEM = """You are a hidden technical observer for a live voice interview.
You do NOT speak to the candidate. Judge the latest answer only.
Return JSON only:
{
  "depth": "superficial" | "partial" | "solid" | "strong",
  "score_hint": 0-100,
  "gaps": ["missing idea 1", "missing idea 2"],
  "probe": "ONE specific spoken follow-up question the interviewer should ask if probing",
  "next_action_hint": "followup" | "next_topic",
  "ack": "short warm ack like Thanks — or Nice. Never Got it. or Understood. (max 6 words)",
  "notes": "one internal line for the interviewer"
}
Rules:
- If the answer is vague / definition-only / incomplete → depth=superficial or partial, next_action_hint=followup,
  probe must dig one level deeper (mechanism, trade-off, edge case, complexity, failure mode).
- Never reveal the optimal solution in probe. Never say "use a hashmap" as the answer.
- If solid/strong and on-topic → next_action_hint=next_topic; probe can be empty.
- Keep probe under 40 words, spoken style.
- PROBE QUALITY — the probe is spoken verbatim, so a vague probe ruins the interview:
  * It must build on THEIR actual words and add a concrete anchor: real inputs/numbers, a named
    alternative, a failure to explain, or a decision to justify.
  * NEVER produce: "Can you explain how X is implemented?", "How would you use X in a project?",
    "What are the key operations of X?", "Explain the difference between X and Y", or any bare
    definition request. Those are automatically wrong.
  * Never repeat or thinly reword anything in already_asked_questions.
  * If you cannot form a specific probe, return probe="" and next_action_hint="next_topic".
- After two probes on the same idea, prefer next_action_hint=next_topic instead of drilling further.
"""


def observe_answer(
    *,
    stage: str,
    role_track: str,
    topics: list[str],
    last_question: str,
    student_message: str,
    code_excerpt: str = "",
    difficulty: str = "intermediate",
    asked_questions: list[str] | None = None,
    probes_on_topic: int = 0,
    interviewer_style: str = "friendly",
) -> dict[str, Any] | None:
    """Fast hidden analysis that steers the live interviewer toward Chakra-style probes."""
    if not llm_configured():
        return None
    raw = chat(
        [
            {"role": "system", "content": OBSERVER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": stage,
                        "role_track": role_track,
                        "topics": (topics or [])[:8],
                        "difficulty": difficulty,
                        "last_interviewer_question": (last_question or "")[:500],
                        "candidate_just_said": (student_message or "")[:900],
                        "code_excerpt": (code_excerpt or "")[:1200] or None,
                        "already_asked_questions": [
                            str(q)[:200] for q in (asked_questions or [])[-8:]
                        ],
                        "probes_already_on_this_idea": int(probes_on_topic or 0),
                    }
                ),
            },
        ],
        temperature=0.25,
        max_tokens=220,
        timeout=12.0,
    )
    data = _extract_json(raw or "")
    if not data:
        return None
    depth = str(data.get("depth") or "partial").strip().lower()
    if depth not in {"superficial", "partial", "solid", "strong"}:
        depth = "partial"
    action = str(data.get("next_action_hint") or "followup").strip().lower()
    if action not in {"followup", "next_topic"}:
        action = "followup"
    try:
        score_hint = float(data.get("score_hint", 50))
    except Exception:
        score_hint = 50.0
    gaps = data.get("gaps") if isinstance(data.get("gaps"), list) else []
    gaps = [str(g)[:120] for g in gaps[:4] if str(g).strip()]
    probe = str(data.get("probe") or "").strip()[:280]
    if probe and is_vague_question(probe):
        # A generic probe is worse than no probe — let the interviewer turn craft
        # its own follow-up under the question-quality rules.
        probe = ""
    return {
        "depth": depth,
        "score_hint": max(0.0, min(100.0, score_hint)),
        "gaps": gaps,
        "probe": probe,
        "next_action_hint": action,
        "ack": str(data.get("ack") or default_ack(interviewer_style)).strip()[:48],
        "notes": str(data.get("notes") or "").strip()[:220],
    }


SCORER_SYSTEM = """You score a live technical-interview answer. You do NOT interview and you do NOT write a question.
Return JSON only:
{"score":0-100,"next_action":"followup|next_topic|move_to_coding","topic_tag":"short topic","depth":"superficial|partial|solid|strong","cue":"hidden coach: stay on X or switch to Y"}
Rules:
- score honestly; thin or off-topic answers ≤40.
- followup if they were shallow and the same topic still has room.
- next_topic if the answer was solid or two probes already happened.
- move_to_coding only when coding_enabled is true AND they are clearly ready to leave conceptual Q&A.
- cue is a coach note for the voice model — never a question to read aloud, never a score.
- Stay on faculty_topics. English only. Max 24 words in cue.
"""


def score_turn(
    *,
    stage: str,
    role_track: str,
    topics: list[str],
    last_question: str,
    student_message: str,
    asked_questions: list[str] | None = None,
    briefing: str = "",
    current_topic: str = "",
    include_coding: bool = True,
    probes_on_topic: int = 0,
) -> dict[str, Any] | None:
    """One cheap JSON call: score + next_action + topic cue. Realtime speaks the question."""
    if not llm_configured():
        return None
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()][:8]
    raw = chat(
        [
            {"role": "system", "content": SCORER_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": stage,
                        "role_track": role_track,
                        "coding_enabled": bool(include_coding),
                        "faculty_topics": topic_list,
                        "current_topic": (current_topic or "")[:80],
                        "faculty_briefing": " ".join((briefing or "").split())[:400],
                        "last_interviewer_question": (last_question or "")[:280],
                        "candidate_just_said": (student_message or "")[:700],
                        "already_asked": [str(q)[:120] for q in (asked_questions or [])[-6:]],
                        "probes_already_on_this_idea": int(probes_on_topic or 0),
                    }
                ),
            },
        ],
        temperature=0.2,
        max_tokens=160,
        timeout=10.0,
    )
    data = _extract_json(raw or "")
    if not data:
        return None
    action = str(data.get("next_action") or "followup").strip().lower()
    if action not in {"followup", "next_topic", "move_to_coding"}:
        action = "followup"
    depth = str(data.get("depth") or "partial").strip().lower()
    if depth not in {"superficial", "partial", "solid", "strong"}:
        depth = "partial"
    try:
        score = float(data.get("score", 50))
    except Exception:
        score = 50.0
    topic_tag = str(data.get("topic_tag") or current_topic or (topic_list[0] if topic_list else "")).strip()[:80]
    cue = " ".join(str(data.get("cue") or "").split())[:180]
    if not cue:
        cue = f"Stay on: {topic_tag}." if topic_tag else "Probe one level deeper, then one short question."
    return {
        "reply": "",
        "cue": cue,
        "score": max(0.0, min(100.0, score)),
        "next_action": action,
        "topic_tag": topic_tag,
        "hint_level": 0,
        "depth": depth,
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
    for t in transcript[-8:]:
        role = "Interviewer" if t.get("role") == "assistant" else "Candidate"
        history_lines.append(f"{role}: {str(t.get('content', ''))[:360]}")
    history = "\n".join(history_lines) if history_lines else "(interview just started)"

    asked = ctx.get("asked_topics") or []
    resume = (ctx.get("resume_text") or "")[:500]
    dossier = ctx.get("resume_dossier") or {}
    must_ask = ctx.get("must_ask_next") or {}
    weak = is_weak_answer(student_message)
    incomplete = looks_incomplete_answer(student_message)
    name = "NexAI"
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
            f"{briefing[:4500]}"
        )
    difficulty = str(ctx.get("difficulty") or "intermediate")
    pace = str(ctx.get("pace") or "standard")
    question_mix = str(ctx.get("question_mix") or "conceptual")
    followup_depth = str(ctx.get("followup_depth") or "moderate")
    avoid_topics = str(ctx.get("avoid_topics") or "").strip()
    extras.extend(
        [
            "Your spoken name is always NexAI. Introduce yourself only as NexAI — "
            "never use a custom profile title as your name.",
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
            "Do not fall back to generic hash-map / Big-O openers unless the briefing asks for them. "
            "Each next_topic question must name a focus topic and ask for a concrete example, "
            "trade-off, edge case, or failure mode matching difficulty="
            + difficulty
            + "."
        )
    allowed_topics = [str(t).strip() for t in (ctx.get("allowed_topics") or []) if str(t).strip()]
    focus_topic = str(ctx.get("focus_topic") or "").strip()
    if allowed_topics and not resume_only:
        extras.append(
            "TOPIC LOCK (HARD CONSTRAINT): every question must sit inside this list — "
            + ", ".join(allowed_topics)
            + ". Never ask about a topic outside the list. Specifically: do NOT ask about time/space "
            "complexity, Big-O, hash maps, arrays, or other DSA topics unless they appear in the list."
        )
        if focus_topic:
            extras.append(
                f"FOCUS TOPIC FOR A next_topic QUESTION: {focus_topic}. "
                "Name it explicitly in the question. (Follow-ups stay on the previous topic.)"
            )
    if not bool(ctx.get("resume_questions_allowed", True)) and not resume_only:
        extras.append(
            "RESUME IS OFF-LIMITS: do not ask about their resume, CV, listed projects, internships, "
            "or past companies. Ask only from the topic list above."
        )
    observer = ctx.get("observer") if isinstance(ctx.get("observer"), dict) else None
    if observer:
        extras.append(
            "TECHNICAL OBSERVER (hidden — MUST STEER THIS TURN):\n"
            f"depth={observer.get('depth')}; score_hint={observer.get('score_hint')}; "
            f"next_action_hint={observer.get('next_action_hint')}; "
            f"gaps={observer.get('gaps')}; notes={observer.get('notes')}.\n"
            f"Preferred probe: {observer.get('probe') or '(none)'}\n"
            f"Preferred short ack: {observer.get('ack') or default_ack(style)}\n"
            "If next_action_hint=followup: reply = short ack + the preferred probe "
            "(paraphrase OK, keep the same depth of probe). next_action MUST be followup.\n"
            "If next_action_hint=next_topic: brief ack then invent a NEW on-topic question; "
            "do not re-ask the same stem."
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
        + (
            f"A next_topic question MUST be about focus_topic ({focus_topic}) and must stay inside "
            "allowed_topics. Never substitute complexity/hash-map/DSA topics that are not listed. "
            if allowed_topics and focus_topic
            else ""
        )
        + f"This turn's format hint: {format_hint} "
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
        "allowed_topics": allowed_topics or None,
        "focus_topic": focus_topic or None,
        "already_asked_questions": [
            str(q)[:220] for q in (ctx.get("asked_questions") or [])[-8:]
        ],
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
        "observer": observer,
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
        max_tokens=420 if resume_only else (320 if dynamic_ok else (280 if suggested_format in {"predict", "debug", "complexity"} else 220)),
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

    reply = strip_spoken_meta(str(data["reply"]).strip())
    if not reply:
        return None
    allowed_topics = [str(t).strip() for t in (ctx.get("allowed_topics") or []) if str(t).strip()]
    if is_vague_question(reply) or is_off_lock_dsa(reply, allowed_topics):
        # Orchestrator will replace with a concrete on-topic question.
        action = "next_topic"
    tag = str(data.get("topic_tag") or "").strip()
    if tag.lower() in {"self-introduction", "opening", "intro", "self introduction"}:
        tag = str(ctx.get("focus_topic") or "")[:80]
    return {
        "reply": reply,
        "score": score,
        "next_action": action,
        "topic_tag": tag,
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
    focus_topic: str = "",
    resume_questions_allowed: bool = True,
) -> dict[str, Any] | None:
    """Generate the opening conceptual question (anchored to question graph when provided)."""
    if not llm_configured():
        return None
    node = question_node or {}
    stem = str(node.get("spoken_now") or node.get("stem") or "").strip()
    name = "NexAI"
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
            f"{briefing[:4500]}"
        )
    system += (
        "\n\nYour spoken name is always NexAI. Introduce yourself only as NexAI, never as a custom profile title. "
        f"dynamic_question_ok={str(dynamic_ok).lower()}. "
        f"Style={interviewer_style}. Difficulty={difficulty}. Pace={pace}. "
        f"Question mix={question_mix}. Follow-up depth={followup_depth}. "
        f"suggested_format={suggested_format}."
    )
    if avoid_topics:
        system += f" Do NOT ask about: {avoid_topics[:400]}."
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()][:12]
    focus_topic = str(focus_topic or "").strip() or (topic_list[0] if topic_list else "")
    if topic_list and not resume_only:
        numbered = " → ".join(f"{i}. {t}" for i, t in enumerate(topic_list, 1))
        system += (
            "\n\nTOPIC LOCK (HARD CONSTRAINT): cover this AGENDA in order — "
            + numbered
            + ". The opening question MUST be item 1"
            + (f" ({focus_topic})" if focus_topic else "")
            + ". Do NOT skip ahead. Do NOT ask about time/space complexity, Big-O, hash maps, "
            "or other DSA topics unless they appear in the list."
        )
    if not resume_questions_allowed and not resume_only:
        system += (
            " RESUME IS OFF-LIMITS: do not ask about their resume, CV, listed projects, internships, "
            "or past companies."
        )
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
            + (
                f"Open on this topic: {focus_topic}. Name it in the question. "
                if focus_topic
                else "Pick ONE focus topic from the topics list (or briefing). Name it in the question. "
            )
            + "Ask for a concrete example, trade-off, or failure mode matching difficulty="
            f"{difficulty}. "
            "Do NOT default to generic hash-map / Big-O / sorted-array unless those topics are listed. "
            "Do not use bank stems."
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
                        "allowed_topics": topic_list or None,
                        "focus_topic": focus_topic or None,
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
                            "You are NexAI. Give ONE clean spoken intro only once "
                            f"(greet, say you are NexAI, {coding_line}), then ask ONE specific technical question. "
                            "Do not stack two greetings. Do not add random filler or a second intro. "
                            "Never introduce yourself with a custom interviewer profile title. "
                            f"{ask_rule} "
                            "next_action=next_topic. hint_level=0."
                        ),
                        "transcript": "(start)",
                    }
                ),
            },
        ],
        temperature=0.55 if resume_only else (0.7 if dynamic_ok else 0.55),
        max_tokens=480 if dynamic_ok else 420,
        timeout=30.0,
    )
    data = _extract_json(raw or "")
    if not data or not str(data.get("reply") or "").strip():
        if raw and not data:
            _set_error(f"Could not parse opening JSON: {(raw or '')[:200]}")
        return None
    tag = str(data.get("topic_tag") or node.get("skill") or node.get("question_id") or "").strip()
    if tag.lower() in {"self-introduction", "opening", "intro", "self introduction"}:
        tag = focus_topic or (topic_list[0] if topic_list else tag)
    reply = strip_spoken_meta(str(data["reply"]).strip())
    if not reply or is_vague_question(reply, min_words=8) or is_off_lock_dsa(reply, topic_list):
        return None
    if resume_only and must_ask_next and must_ask_next.get("question"):
        planned = str(must_ask_next.get("question") or "").strip()
        if planned and "?" not in reply:
            reply = (reply.rstrip(" .") + ". " + planned).strip()
    return {
        "reply": reply,
        "score": 0,
        "next_action": "next_topic",
        "topic_tag": tag or focus_topic or "opening",
        "hint_level": 0,
        "question_id": "" if dynamic_ok else (node.get("question_id") or ""),
    }


_STYLE_TONE = {
    "friendly": "warm and encouraging, still rigorous",
    "strict": "crisp and demanding, never harsh or mocking",
    "brief": "concise and businesslike, still polite",
    "socratic": "curious and Socratic, never lecturing",
    "supportive": "calm and supportive",
    "panel": "professional hiring-bar, evidence-seeking",
}


def _quality_rules(topic_list: list[str], focus_topic: str, style: str) -> str:
    tone = _STYLE_TONE.get((style or "friendly").lower(), "professional and clear")
    lines = [
        f"Tone: {tone}. Hold this tone even when the candidate struggles.",
        "The question must be CONCRETE: name the topic and anchor it to a specific scenario, "
        "real inputs/numbers, a named alternative to compare, or an observed behaviour to explain.",
        "BANNED shapes: 'how would you use X in a project', 'can you explain how X is implemented', "
        "'walk me through how you'd approach it', 'what are the key operations', bare "
        "'difference between X and Y', bare 'tell me about X', or any textbook-definition request.",
        "One question only, answerable in about 20 seconds of speech, max 28 spoken words.",
        "Never repeat or thinly reword a question in already_asked_questions.",
    ]
    if topic_list:
        lines.append(
            "TOPIC LOCK (HARD): the question must sit inside — " + ", ".join(topic_list)
            + ". Do NOT ask about Big-O, complexity, hash maps, or other DSA topics unless listed."
        )
    if focus_topic:
        lines.append(f"Ask about this topic: {focus_topic}. Name it naturally in the question.")
    return "\n- ".join(lines)


def topic_question(
    *,
    role_track: str,
    topics: list[str],
    focus_topic: str = "",
    difficulty: str = "intermediate",
    interviewer_style: str = "friendly",
    interviewer_briefing: str = "",
    asked_questions: list[str] | None = None,
    transcript: list[dict[str, str]] | None = None,
    bridge_hint: str = "",
    resume_questions_allowed: bool = True,
) -> dict[str, Any] | None:
    """
    Generate ONE specific next-topic question.

    Used whenever the engine (not the model) decides to change topic, so the
    candidate never hears a bare "let's move on" with no real question.
    """
    if not llm_configured():
        return None
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()][:12]
    focus = str(focus_topic or "").strip() or (topic_list[0] if topic_list else "")
    system = (
        INTERVIEWER_SYSTEM
        + "\n\nThis call produces ONE next-topic question only.\n- "
        + _quality_rules(topic_list, focus, interviewer_style)
    )
    if interviewer_briefing:
        system += (
            "\n\nFACULTY BRIEFING / CUSTOM INTERVIEWER RULES (primary guide):\n"
            f"{interviewer_briefing[:3500]}"
        )
    if not resume_questions_allowed:
        system += (
            "\n\nRESUME IS OFF-LIMITS: do not ask about their resume, CV, listed projects, "
            "internships, or past companies."
        )
    history = [
        ("Interviewer" if t.get("role") == "assistant" else "Candidate")
        + ": "
        + str(t.get("content", ""))[:240]
        for t in (transcript or [])[-6:]
    ]
    raw = chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "qa",
                        "role_track": role_track,
                        "allowed_topics": topic_list or None,
                        "focus_topic": focus or None,
                        "difficulty": difficulty,
                        "already_asked_questions": [
                            str(q)[:220] for q in (asked_questions or [])[-8:]
                        ],
                        "transcript_tail": history,
                        "stage_instructions": (
                            "The previous topic is finished. Say a SHORT natural bridge "
                            + (f"(something like: {bridge_hint}) " if bridge_hint else "")
                            + "of at most 8 words, then ask ONE new concrete question. "
                            "Never say the candidate failed, never mention scores or grading. "
                            "next_action=next_topic. hint_level=0."
                        ),
                    }
                ),
            },
        ],
        temperature=0.7,
        max_tokens=260,
        timeout=20.0,
    )
    data = _extract_json(raw or "")
    reply = strip_spoken_meta(str((data or {}).get("reply") or "").strip())
    if (
        not reply
        or "?" not in reply
        or is_vague_question(reply, min_words=8)
        or is_off_lock_dsa(reply, topic_list)
    ):
        return None
    return {
        "reply": reply,
        "topic_tag": str((data or {}).get("topic_tag") or focus or "")[:80],
    }


def respecify_question(
    *,
    last_question: str,
    candidate_reply: str,
    topics: list[str],
    focus_topic: str = "",
    difficulty: str = "intermediate",
    interviewer_style: str = "friendly",
    interviewer_briefing: str = "",
    asked_questions: list[str] | None = None,
) -> str | None:
    """
    Rewrite the last question concretely after the candidate said it was unclear.

    This is the repair path for "I didn't get your question" / "that question is absurd".
    """
    if not llm_configured():
        return None
    topic_list = [str(t).strip() for t in (topics or []) if str(t).strip()][:12]
    focus = str(focus_topic or "").strip() or (topic_list[0] if topic_list else "")
    system = (
        INTERVIEWER_SYSTEM
        + "\n\nThis call REPAIRS one unclear question.\n- "
        + _quality_rules(topic_list, focus, interviewer_style)
    )
    if interviewer_briefing:
        system += (
            "\n\nFACULTY BRIEFING / CUSTOM INTERVIEWER RULES (primary guide):\n"
            f"{interviewer_briefing[:2500]}"
        )
    raw = chat(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "stage": "qa",
                        "your_previous_question": (last_question or "")[:600],
                        "candidate_objection": (candidate_reply or "")[:400],
                        "allowed_topics": topic_list or None,
                        "focus_topic": focus or None,
                        "difficulty": difficulty,
                        "already_asked_questions": [
                            str(q)[:220] for q in (asked_questions or [])[-6:]
                        ],
                        "stage_instructions": (
                            "The candidate says your previous question was unclear or did not make "
                            "sense — they are right. Own it in at most five words, then ask the SAME "
                            "competency as a fully concrete question: give a tiny scenario with real "
                            "inputs or a named choice they must justify. Do not defend the old wording, "
                            "do not shorten it into something vaguer, do not change topic, do not say "
                            "anything about scoring. next_action=followup. hint_level=1."
                        ),
                    }
                ),
            },
        ],
        temperature=0.6,
        max_tokens=240,
        timeout=20.0,
    )
    data = _extract_json(raw or "")
    reply = strip_spoken_meta(str((data or {}).get("reply") or "").strip())
    if (
        not reply
        or "?" not in reply
        or is_vague_question(reply, min_words=8)
        or is_off_lock_dsa(reply, topic_list)
    ):
        return None
    return reply


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
