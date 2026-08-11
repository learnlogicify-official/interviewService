"""Problem bank loader."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "problems.json"


@lru_cache
def load_problems() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def get_problem(problem_id: str) -> dict[str, Any] | None:
    for p in load_problems():
        if p["id"] == problem_id:
            return p
    return None


def pick_problem(used_ids: list[str], topics: list[str], prefer: str = "easy") -> dict[str, Any]:
    problems = load_problems()
    topic_set = {t.lower() for t in topics}
    candidates = [
        p
        for p in problems
        if p["id"] not in used_ids and (not topic_set or topic_set.intersection(t.lower() for t in p.get("topics", [])))
    ]
    if not candidates:
        candidates = [p for p in problems if p["id"] not in used_ids] or problems
    ordered = sorted(candidates, key=lambda p: {"easy": 0, "medium": 1, "hard": 2}.get(p.get("difficulty", "medium"), 1))
    if prefer == "medium":
        mid = [p for p in ordered if p.get("difficulty") == "medium"]
        return mid[0] if mid else ordered[0]
    if prefer == "hard":
        hard = [p for p in ordered if p.get("difficulty") == "hard"]
        return hard[-1] if hard else ordered[-1]
    return ordered[0]


def public_problem(problem: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": problem["id"],
        "title": problem["title"],
        "difficulty": problem["difficulty"],
        "topics": problem.get("topics", []),
        "prompt": problem["prompt"],
        "starter_code": problem.get("starter_code", ""),
        "sample_tests": problem.get("sample_tests", []),
    }
