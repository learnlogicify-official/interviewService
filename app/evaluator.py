"""Heuristic + optional LLM evaluation helpers."""

from __future__ import annotations

import re
from typing import Any

import httpx

from app.config import get_settings


TECH_BANK = [
    {
        "id": "hashmap",
        "question": "In a coding interview, when would you prefer a hash map over a sorted array for lookups? What are the trade-offs?",
        "keywords": ["o(1)", "average", "hash", "unordered", "space", "collision", "sorted", "log n"],
    },
    {
        "id": "complexity",
        "question": "Explain Big-O of time and space. Give an example of O(n) time and O(1) extra space.",
        "keywords": ["big-o", "linear", "constant", "input size", "worst", "space", "time"],
    },
    {
        "id": "stack",
        "question": "What is a stack? Name one classic problem where a stack is the natural choice.",
        "keywords": ["lifo", "push", "pop", "parentheses", "dfs", "undo", "call stack"],
    },
    {
        "id": "recursion",
        "question": "How does recursion use the call stack? When would you convert recursion to iteration?",
        "keywords": ["call stack", "base case", "overflow", "iteration", "tail", "depth"],
    },
    {
        "id": "dbms",
        "question": "What is an index in a database? When can an index hurt write performance?",
        "keywords": ["index", "b-tree", "lookup", "write", "update", "disk", "query"],
    },
    {
        "id": "oops",
        "question": "Explain encapsulation vs abstraction with a short software example.",
        "keywords": ["encapsulation", "abstraction", "hide", "interface", "class", "private"],
    },
]


def score_keywords(answer: str, keywords: list[str]) -> float:
    text = answer.lower()
    if len(text.strip()) < 20:
        return 25.0
    hits = sum(1 for k in keywords if k.lower() in text)
    ratio = hits / max(1, len(keywords))
    length_bonus = 10 if len(text.split()) >= 40 else 0
    return min(100.0, 35 + ratio * 55 + length_bonus)


def score_idea(answer: str, problem: dict[str, Any]) -> dict[str, Any]:
    text = answer.lower()
    good = problem.get("good_ideas", [])
    weak = problem.get("weak_ideas", [])
    good_hits = sum(1 for g in good if any(tok in text for tok in g.lower().split()[:3]))
    weak_hits = sum(1 for w in weak if any(tok in text for tok in w.lower().split()[:3]))
    score = 40.0 + good_hits * 20 - weak_hits * 15
    if "o(n)" in text or "linear" in text:
        score += 10
    if "o(n^2)" in text or "nested" in text:
        score += 5
    score = max(0.0, min(100.0, score))
    accepted = score >= 55 and len(text.split()) >= 25
    feedback = (
        "Solid approach — you may start coding."
        if accepted
        else "Push further: name the data structure, complexity, and edge cases before coding."
    )
    return {"score": score, "accepted": accepted, "feedback": feedback}


def pick_code_excerpt(code: str) -> str:
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return ""
    # Prefer a non-trivial middle chunk.
    start = max(0, len(lines) // 3)
    chunk = lines[start : start + 4]
    return "\n".join(chunk)


def score_explanation(answer: str, excerpt: str) -> float:
    text = answer.lower()
    tokens = re.findall(r"[a-z_]{3,}", excerpt.lower())
    hits = sum(1 for t in set(tokens) if t in text)
    base = 40 + min(40, hits * 8)
    if len(text.split()) < 15:
        base -= 15
    return max(0.0, min(100.0, float(base)))


async def maybe_llm_polish(system: str, user: str) -> str | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "temperature": 0.4,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
