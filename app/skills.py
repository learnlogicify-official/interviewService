"""Track-aware skill graph for adaptive interviewing."""

from __future__ import annotations

from typing import Any


# Default skill trees per role track. Scores are 0–1 confidence.
TRACK_SKILLS: dict[str, dict[str, Any]] = {
    "sde_intern": {
        "dsa": {"label": "DSA", "children": {"arrays": 0.5, "hashmap": 0.5, "stacks": 0.5, "complexity": 0.5}},
        "oop": {"label": "OOP", "children": {"encapsulation": 0.5, "polymorphism": 0.5}},
        "sql": {"label": "SQL", "children": {"indexes": 0.5, "joins": 0.5}},
        "projects": {
            "label": "Projects & experience",
            "children": {"architecture": 0.5, "debugging": 0.5, "tradeoffs": 0.5},
        },
        "systems": {
            "label": "Systems",
            "children": {"caching": 0.5, "queues": 0.5, "concurrency": 0.5},
        },
        "coding": {"label": "Coding", "children": {"correctness": 0.5, "edge_cases": 0.5, "style": 0.5}},
        "communication": {"label": "Communication", "children": {"clarity": 0.5, "structure": 0.5}},
    },
    "frontend": {
        "javascript": {"label": "JavaScript", "children": {"closures": 0.5, "async": 0.5, "dom": 0.5}},
        "react": {"label": "React", "children": {"hooks": 0.5, "state": 0.5, "perf": 0.5}},
        "css": {"label": "CSS", "children": {"layout": 0.5, "responsive": 0.5}},
        "projects": {
            "label": "Projects & experience",
            "children": {"architecture": 0.5, "debugging": 0.5, "tradeoffs": 0.5},
        },
        "coding": {"label": "Coding", "children": {"correctness": 0.5, "edge_cases": 0.5}},
        "communication": {"label": "Communication", "children": {"clarity": 0.5, "structure": 0.5}},
    },
    "backend": {
        "apis": {"label": "APIs", "children": {"rest": 0.5, "auth": 0.5, "idempotency": 0.5}},
        "databases": {"label": "Databases", "children": {"sql": 0.5, "indexes": 0.5, "transactions": 0.5}},
        "systems": {"label": "Systems", "children": {"caching": 0.5, "queues": 0.5, "scalability": 0.5}},
        "projects": {
            "label": "Projects & experience",
            "children": {"architecture": 0.5, "debugging": 0.5, "tradeoffs": 0.5},
        },
        "coding": {"label": "Coding", "children": {"correctness": 0.5, "edge_cases": 0.5}},
        "communication": {"label": "Communication", "children": {"clarity": 0.5, "structure": 0.5}},
    },
    "ai_ml": {
        "ml": {"label": "ML", "children": {"supervised": 0.5, "eval": 0.5, "overfitting": 0.5}},
        "llm": {"label": "LLMs", "children": {"prompting": 0.5, "rag": 0.5, "safety": 0.5}},
        "python": {"label": "Python", "children": {"numpy": 0.5, "data": 0.5}},
        "projects": {
            "label": "Projects & experience",
            "children": {"architecture": 0.5, "debugging": 0.5, "tradeoffs": 0.5},
        },
        "coding": {"label": "Coding", "children": {"correctness": 0.5, "edge_cases": 0.5}},
        "communication": {"label": "Communication", "children": {"clarity": 0.5, "structure": 0.5}},
    },
    "resume_deep": {
        "projects": {"label": "Projects", "children": {"ownership": 0.5, "architecture": 0.5, "tradeoffs": 0.5}},
        "experience": {"label": "Experience", "children": {"internships": 0.5, "impact": 0.5, "collaboration": 0.5}},
        "stack": {"label": "Claimed stack", "children": {"depth": 0.5, "debugging": 0.5, "fundamentals": 0.5}},
        "communication": {"label": "Communication", "children": {"clarity": 0.5, "structure": 0.5}},
    },
}


TOPIC_ALIASES: dict[str, tuple[str, str]] = {
    "hashmap": ("dsa", "hashmap"),
    "hash map": ("dsa", "hashmap"),
    "arrays": ("dsa", "arrays"),
    "array": ("dsa", "arrays"),
    "inversion": ("dsa", "arrays"),
    "merge sort": ("dsa", "complexity"),
    "mergesort": ("dsa", "complexity"),
    "circular": ("dsa", "arrays"),
    "queue": ("dsa", "stacks"),
    "stack": ("dsa", "stacks"),
    "stacks": ("dsa", "stacks"),
    "complexity": ("dsa", "complexity"),
    "big-o": ("dsa", "complexity"),
    "oop": ("oop", "encapsulation"),
    "encapsulation": ("oop", "encapsulation"),
    "polymorphism": ("oop", "polymorphism"),
    "abstract": ("oop", "polymorphism"),
    "interface": ("oop", "polymorphism"),
    "inheritance": ("oop", "polymorphism"),
    "java": ("oop", "encapsulation"),
    "class": ("oop", "encapsulation"),
    "sql": ("sql", "joins"),
    "index": ("sql", "indexes"),
    "indexes": ("sql", "indexes"),
    "join": ("sql", "joins"),
    "react": ("react", "hooks"),
    "hooks": ("react", "hooks"),
    "javascript": ("javascript", "closures"),
    "async": ("javascript", "async"),
    "rest": ("apis", "rest"),
    "auth": ("apis", "auth"),
    "cache": ("systems", "caching"),
    "caching": ("systems", "caching"),
    "redis": ("systems", "caching"),
    "sidekiq": ("systems", "queues"),
    "worker": ("systems", "queues"),
    "workers": ("systems", "queues"),
    "concurrency": ("systems", "concurrency"),
    "concurrent": ("systems", "concurrency"),
    "compile": ("systems", "concurrency"),
    "compilation": ("systems", "concurrency"),
    "load": ("systems", "concurrency"),
    "lms": ("projects", "architecture"),
    "project": ("projects", "architecture"),
    "projects": ("projects", "architecture"),
    "resume": ("projects", "architecture"),
    "internship": ("projects", "tradeoffs"),
    "franchise": ("projects", "tradeoffs"),
    "rag": ("llm", "rag"),
    "llm": ("llm", "prompting"),
}


def default_graph(role_track: str) -> dict[str, Any]:
    track = role_track if role_track in TRACK_SKILLS else "sde_intern"
    # Deep copy-ish via json-less manual copy.
    src = TRACK_SKILLS[track]
    out: dict[str, Any] = {}
    for key, node in src.items():
        out[key] = {
            "label": node["label"],
            "children": {k: float(v) for k, v in node["children"].items()},
        }
    return out


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _fallback_skill(graph: dict[str, Any]) -> tuple[str, str]:
    """Route unknown tags to a topic bucket — never the communication clarity leaf."""
    for parent, child in (
        ("projects", "architecture"),
        ("stack", "depth"),
        ("systems", "concurrency"),
        ("dsa", "complexity"),
        ("oop", "encapsulation"),
    ):
        if parent in graph and child in (graph[parent].get("children") or {}):
            return parent, child
    parent = "communication" if "communication" in graph else next(iter(graph))
    kids = graph[parent].get("children") or {}
    child = "structure" if "structure" in kids else next(iter(kids), "clarity")
    return parent, child


def update_skill(
    graph: dict[str, Any],
    *,
    topic_tag: str,
    score_0_100: float,
    evidence: str = "",
    weight: float = 0.35,
) -> dict[str, Any]:
    """Blend a 0–100 answer score into the matching skill leaf."""
    tag = (topic_tag or "").strip().lower()
    if not tag or not graph:
        return graph

    parent, child = None, None
    for alias, pair in TOPIC_ALIASES.items():
        if alias in tag or tag in alias:
            parent, child = pair
            break
    if parent is None:
        # Try direct parent/child names.
        for p, node in graph.items():
            if str(p).startswith("_"):
                continue
            if p in tag:
                parent = p
                kids = list(node.get("children", {}).keys())
                child = kids[0] if kids else None
                for k in kids:
                    if k in tag:
                        child = k
                        break
                break
    if parent is None or parent not in graph:
        parent, child = _fallback_skill(graph)
    elif child not in (graph.get(parent, {}).get("children") or {}):
        parent, child = _fallback_skill(graph)

    kids = graph[parent].setdefault("children", {})
    if child not in kids:
        child = next(iter(kids)) if kids else "general"
        kids.setdefault(child, 0.5)

    w = max(0.15, min(0.9, float(weight)))
    target = _clamp(score_0_100 / 100.0)
    prev = float(kids.get(child, 0.5))
    # Exponential moving average — recent answers matter more.
    kids[child] = _clamp(prev * (1.0 - w) + target * w)

    touched = graph.setdefault("_touched", [])
    key = f"{parent}.{child}"
    if key not in touched:
        touched.append(key)
    graph["_touched"] = touched[-40:]

    evidence_list = graph.setdefault("_evidence", [])
    if evidence:
        evidence_list.append(
            {
                "skill": key,
                "score": round(score_0_100, 1),
                "note": evidence[:240],
            }
        )
        graph["_evidence"] = evidence_list[-40:]
    return graph


def weakest_skills(graph: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent, node in graph.items():
        if parent.startswith("_"):
            continue
        for child, val in (node.get("children") or {}).items():
            rows.append(
                {
                    "parent": parent,
                    "child": child,
                    "label": f"{node.get('label', parent)} / {child}",
                    "score": round(float(val) * 100, 1),
                }
            )
    rows.sort(key=lambda r: r["score"])
    return rows[:limit]


def strongest_skills(graph: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    rows = weakest_skills(graph, limit=100)
    rows.sort(key=lambda r: -r["score"])
    return rows[:limit]


def summarize_for_llm(graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "weakest": weakest_skills(graph, 4),
        "strongest": strongest_skills(graph, 3),
        "evidence_tail": (graph.get("_evidence") or [])[-6:],
    }
