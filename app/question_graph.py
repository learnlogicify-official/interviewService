"""Curated competency question graph for adaptive interviewing.

Nodes are track-scoped. Selection prefers weakest skills, respects difficulty
ceiling, and surfaces follow-ups / deep probes instead of free-form LLM invention.
"""

from __future__ import annotations

from typing import Any

from app import skills as skill_graph


# Difficulty: 1 = warmup, 5 = hard. Engine raises/lowers ceiling from evidence.
QuestionNode = dict[str, Any]

QUESTION_BANK: list[QuestionNode] = [
    # —— SDE intern / shared DSA ——
    {
        "id": "sde.hashmap.lookups",
        "tracks": ["sde_intern", "backend"],
        "skill": ("dsa", "hashmap"),
        "difficulty": 2,
        "stem": (
            "When would you prefer a hash map over a sorted array for lookups, "
            "and what trade-offs would you accept?"
        ),
        "keywords": ["o(1)", "average", "hash", "unordered", "space", "collision", "log n"],
        "followups": [
            "What happens under heavy collisions, and how would you detect that in practice?",
            "How does that change if you need ordered iteration?",
        ],
        "deep_probes": [
            "If two keys collide constantly, what breaks first — latency, memory, or correctness?",
        ],
        "rubric": {
            "strong": "Names average O(1) vs O(log n), space cost, and at least one collision/order caveat.",
            "weak": "Only says hash map is faster with no complexity or trade-off.",
        },
    },
    {
        "id": "sde.arrays.twopointer",
        "tracks": ["sde_intern"],
        "skill": ("dsa", "arrays"),
        "difficulty": 2,
        "stem": (
            "Give one problem where two pointers on an array beat a nested loop, "
            "and why the time complexity improves."
        ),
        "keywords": ["two pointer", "sorted", "o(n)", "left", "right", "window"],
        "followups": [
            "Does the array need to be sorted first? What does that do to overall complexity?",
        ],
        "deep_probes": [
            "Walk through duplicates — where does your pointer movement go wrong?",
        ],
        "rubric": {
            "strong": "Concrete problem + O(n) intuition + sorting caveat if needed.",
            "weak": "Vague 'scan twice' with no complexity story.",
        },
    },
    {
        "id": "sde.complexity.bigo",
        "tracks": ["sde_intern", "backend", "frontend", "ai_ml"],
        "skill": ("dsa", "complexity"),
        "difficulty": 1,
        "stem": (
            "Explain Big-O for time and space. Give an O(n) time and O(1) extra space example."
        ),
        "keywords": ["big-o", "linear", "constant", "input size", "worst", "space", "time"],
        "followups": [
            "What's the difference between best, average, and worst case here?",
        ],
        "deep_probes": [
            "If the input is already sorted, does your Big-O claim still hold?",
        ],
        "rubric": {
            "strong": "Defines growth vs input size and gives a real O(n)/O(1) example.",
            "weak": "Memorizes labels without relating to input size.",
        },
    },
    {
        "id": "sde.stacks.lifo",
        "tracks": ["sde_intern"],
        "skill": ("dsa", "stacks"),
        "difficulty": 2,
        "stem": "What is a stack? Name one classic problem where a stack is the natural choice.",
        "keywords": ["lifo", "push", "pop", "parentheses", "dfs", "undo", "call stack"],
        "followups": [
            "How does the call stack relate to recursion depth?",
        ],
        "deep_probes": [
            "If recursion depth hits a million, what fails first and how would you rewrite it?",
        ],
        "rubric": {
            "strong": "LIFO + concrete problem + optional call-stack link.",
            "weak": "Only says push/pop with no use case.",
        },
    },
    {
        "id": "sde.oop.encapsulation",
        "tracks": ["sde_intern"],
        "skill": ("oop", "encapsulation"),
        "difficulty": 2,
        "stem": "Explain encapsulation versus abstraction with a short software example.",
        "keywords": ["encapsulation", "abstraction", "hide", "interface", "class", "private"],
        "followups": [
            "Where would you put validation — in the constructor, setters, or a factory?",
        ],
        "deep_probes": [
            "If a teammate reaches into private fields via reflection, what design smell does that show?",
        ],
        "rubric": {
            "strong": "Clear distinction + concrete example.",
            "weak": "Uses the words interchangeably.",
        },
    },
    {
        "id": "sde.sql.indexes",
        "tracks": ["sde_intern", "backend"],
        "skill": ("sql", "indexes"),
        "difficulty": 3,
        "stem": "What is a database index, and when can an index hurt write performance?",
        "keywords": ["index", "b-tree", "lookup", "write", "update", "disk", "query"],
        "followups": [
            "Would you index a column that is updated on almost every write? Why or why not?",
        ],
        "deep_probes": [
            "How would you choose between a composite index (a,b) versus two single-column indexes?",
        ],
        "rubric": {
            "strong": "Faster reads vs extra write/maintenance cost.",
            "weak": "Only says indexes make queries faster.",
        },
    },
    {
        "id": "sde.sql.joins",
        "tracks": ["sde_intern", "backend"],
        "skill": ("sql", "joins"),
        "difficulty": 2,
        "stem": "When do you use an INNER JOIN versus a LEFT JOIN? Give a one-line business example.",
        "keywords": ["inner", "left", "null", "match", "outer", "missing"],
        "followups": [
            "What rows disappear with INNER JOIN that LEFT JOIN would keep?",
        ],
        "deep_probes": [
            "How would a missing foreign key change your join choice in a report query?",
        ],
        "rubric": {
            "strong": "Match semantics + concrete example with missing rows.",
            "weak": "Cannot distinguish the joins.",
        },
    },
    # —— Frontend ——
    {
        "id": "fe.js.closures",
        "tracks": ["frontend"],
        "skill": ("javascript", "closures"),
        "difficulty": 2,
        "stem": "What is a closure in JavaScript, and when have you used one intentionally?",
        "keywords": ["closure", "scope", "lexical", "function", "retain", "callback"],
        "followups": [
            "How can closures cause memory leaks in a long-lived SPA?",
        ],
        "deep_probes": [
            "In a loop of listeners, how would a closure capture the wrong index — and how do you fix it?",
        ],
        "rubric": {
            "strong": "Lexical scope + retained variables + real use.",
            "weak": "Vague 'function inside function'.",
        },
    },
    {
        "id": "fe.js.async",
        "tracks": ["frontend"],
        "skill": ("javascript", "async"),
        "difficulty": 3,
        "stem": "How do you handle a chain of async API calls where later calls depend on earlier results?",
        "keywords": ["async", "await", "promise", "then", "error", "race"],
        "followups": [
            "Where do you catch errors — per call or once at the top?",
        ],
        "deep_probes": [
            "If the user navigates away mid-request, how do you cancel or ignore the result?",
        ],
        "rubric": {
            "strong": "await/Promise sequencing + error path.",
            "weak": "Only mentions setTimeout or callbacks.",
        },
    },
    {
        "id": "fe.react.hooks",
        "tracks": ["frontend"],
        "skill": ("react", "hooks"),
        "difficulty": 2,
        "stem": "When would you reach for useEffect, and what dependency-array mistake have you seen?",
        "keywords": ["useeffect", "dependency", "render", "mount", "cleanup"],
        "followups": [
            "What belongs in the cleanup function?",
        ],
        "deep_probes": [
            "Why can an empty dependency array still cause stale state bugs?",
        ],
        "rubric": {
            "strong": "Effect purpose + deps + cleanup awareness.",
            "weak": "Says 'run side effects' with no deps story.",
        },
    },
    {
        "id": "fe.css.layout",
        "tracks": ["frontend"],
        "skill": ("css", "layout"),
        "difficulty": 2,
        "stem": "When do you choose Flexbox over CSS Grid for a layout?",
        "keywords": ["flex", "grid", "one-dimensional", "two-dimensional", "align"],
        "followups": [
            "How would you make that layout responsive without media-query soup?",
        ],
        "deep_probes": [
            "What breaks if a flex child has min-width: auto and long unbreakable text?",
        ],
        "rubric": {
            "strong": "1D vs 2D intuition + one practical caveat.",
            "weak": "Personal preference only.",
        },
    },
    # —— Backend ——
    {
        "id": "be.apis.rest",
        "tracks": ["backend"],
        "skill": ("apis", "rest"),
        "difficulty": 2,
        "stem": "How do you design a REST endpoint that creates a resource and stays idempotent on retries?",
        "keywords": ["idempotent", "post", "put", "key", "retry", "status"],
        "followups": [
            "Where would you store the idempotency key?",
        ],
        "deep_probes": [
            "What status code and body do you return on a duplicate successful retry?",
        ],
        "rubric": {
            "strong": "Idempotency key / PUT semantics + retry story.",
            "weak": "Only says POST creates.",
        },
    },
    {
        "id": "be.apis.auth",
        "tracks": ["backend"],
        "skill": ("apis", "auth"),
        "difficulty": 3,
        "stem": "Compare session cookies and JWT bearer tokens for a browser SPA talking to your API.",
        "keywords": ["jwt", "cookie", "csrf", "expiry", "refresh", "httpOnly"],
        "followups": [
            "How do you rotate or revoke a compromised token?",
        ],
        "deep_probes": [
            "Where do you store the refresh token, and why not localStorage?",
        ],
        "rubric": {
            "strong": "Trade-offs around CSRF, storage, expiry.",
            "weak": "JWT is modern / cookies are old.",
        },
    },
    {
        "id": "be.systems.caching",
        "tracks": ["backend"],
        "skill": ("systems", "caching"),
        "difficulty": 3,
        "stem": "Where would you put a cache in a read-heavy API, and how do you invalidate it safely?",
        "keywords": ["cache", "ttl", "invalidate", "stale", "redis", "hit"],
        "followups": [
            "What is stale-while-revalidate in this context?",
        ],
        "deep_probes": [
            "How do you avoid stampeding a cold cache after expiry?",
        ],
        "rubric": {
            "strong": "Placement + TTL/invalidation + at least one failure mode.",
            "weak": "Just 'use Redis'.",
        },
    },
    {
        "id": "be.db.transactions",
        "tracks": ["backend"],
        "skill": ("databases", "transactions"),
        "difficulty": 3,
        "stem": "When do you need a database transaction across two writes, and what can go wrong without one?",
        "keywords": ["transaction", "atomic", "rollback", "isolation", "commit"],
        "followups": [
            "What isolation level would you pick for a money transfer?",
        ],
        "deep_probes": [
            "How does a long transaction affect lock contention under load?",
        ],
        "rubric": {
            "strong": "Atomicity + concrete failure without TX.",
            "weak": "Transactions are for speed.",
        },
    },
    # —— AI / ML ——
    {
        "id": "ml.supervised.basics",
        "tracks": ["ai_ml"],
        "skill": ("ml", "supervised"),
        "difficulty": 2,
        "stem": "How do you choose between classification and regression for a business metric?",
        "keywords": ["classification", "regression", "label", "continuous", "discrete"],
        "followups": [
            "What label noise would break your choice?",
        ],
        "deep_probes": [
            "If the positive class is 1% of rows, what metric do you refuse to trust?",
        ],
        "rubric": {
            "strong": "Target type + one metric caveat.",
            "weak": "Names models without framing the target.",
        },
    },
    {
        "id": "ml.eval.split",
        "tracks": ["ai_ml"],
        "skill": ("ml", "eval"),
        "difficulty": 2,
        "stem": "Why do we hold out a test set, and when is a single train/test split not enough?",
        "keywords": ["test", "validation", "overfit", "cross", "leakage"],
        "followups": [
            "Where does data leakage usually sneak in?",
        ],
        "deep_probes": [
            "If features are scaled using the full dataset before the split, what did you leak?",
        ],
        "rubric": {
            "strong": "Generalization + leakage or CV awareness.",
            "weak": "Test set is for accuracy only.",
        },
    },
    {
        "id": "ml.llm.rag",
        "tracks": ["ai_ml"],
        "skill": ("llm", "rag"),
        "difficulty": 3,
        "stem": "When is RAG a better fit than fine-tuning for company-specific answers?",
        "keywords": ["rag", "retrieval", "documents", "fine-tune", "fresh", "citation"],
        "followups": [
            "How do you evaluate whether retrieved chunks are actually useful?",
        ],
        "deep_probes": [
            "What fails first when the corpus updates daily but embeddings are stale?",
        ],
        "rubric": {
            "strong": "Fresh knowledge / citations vs weight updates.",
            "weak": "RAG is always better.",
        },
    },
    {
        "id": "ml.llm.prompting",
        "tracks": ["ai_ml"],
        "skill": ("llm", "prompting"),
        "difficulty": 2,
        "stem": "How would you structure a prompt so the model returns strict JSON your service can parse?",
        "keywords": ["json", "schema", "system", "example", "validate"],
        "followups": [
            "What do you do when the model still returns markdown fences?",
        ],
        "deep_probes": [
            "How do you prevent the model from inventing fields not in your schema?",
        ],
        "rubric": {
            "strong": "Schema + examples + validation/retry.",
            "weak": "Just say please return JSON.",
        },
    },
]


def _skill_key(node: QuestionNode) -> str:
    parent, child = node["skill"]
    return f"{parent}.{child}"


def nodes_for_track(role_track: str) -> list[QuestionNode]:
    track = role_track if role_track else "sde_intern"
    matched = [n for n in QUESTION_BANK if track in n.get("tracks", [])]
    if matched:
        return matched
    return [n for n in QUESTION_BANK if "sde_intern" in n.get("tracks", [])]


def get_node(question_id: str | None) -> QuestionNode | None:
    if not question_id:
        return None
    for n in QUESTION_BANK:
        if n["id"] == question_id:
            return n
    return None


def pick_opening(role_track: str, topics: list[str] | None = None) -> QuestionNode:
    nodes = nodes_for_track(role_track)
    topic_blob = " ".join(topics or []).lower()
    # Prefer difficulty 1–2 that matches requested topics.
    ranked = sorted(
        nodes,
        key=lambda n: (
            0 if any(t and (t in _skill_key(n) or t in n["stem"].lower()) for t in (topics or [])) else 1,
            abs(int(n.get("difficulty", 2)) - 2),
            n["id"],
        ),
    )
    if topic_blob:
        for n in ranked:
            if any(t and t in (_skill_key(n) + n["stem"].lower()) for t in (topics or [])):
                return n
    return ranked[0] if ranked else QUESTION_BANK[0]


def pick_next(
    *,
    role_track: str,
    graph: dict[str, Any],
    asked_ids: list[str],
    difficulty_ceiling: int = 3,
    prefer_skill: tuple[str, str] | None = None,
) -> QuestionNode | None:
    """Choose next unused node: weakest skill first, then within difficulty ceiling."""
    nodes = nodes_for_track(role_track)
    used = set(asked_ids or [])
    candidates = [
        n
        for n in nodes
        if n["id"] not in used and int(n.get("difficulty", 2)) <= max(1, int(difficulty_ceiling))
    ]
    if not candidates:
        candidates = [n for n in nodes if n["id"] not in used]
    if not candidates:
        return None

    weak = skill_graph.weakest_skills(graph, limit=8)
    weak_keys = {f"{w['parent']}.{w['child']}": float(w["score"]) for w in weak}

    def score_row(n: QuestionNode) -> tuple[float, int, str]:
        sk = _skill_key(n)
        parent, child = n["skill"]
        leaf = 50.0
        node = graph.get(parent) or {}
        kids = node.get("children") or {}
        if child in kids:
            leaf = float(kids[child]) * 100.0
        elif sk in weak_keys:
            leaf = weak_keys[sk]
        prefer_boost = -30.0 if prefer_skill and n["skill"] == prefer_skill else 0.0
        # Lower leaf score = weaker = higher priority (sort ascending on first key).
        return (leaf + prefer_boost, int(n.get("difficulty", 2)), n["id"])

    candidates.sort(key=score_row)
    return candidates[0]


def spoken_prompt(node: QuestionNode, *, hint_level: int = 0, followup_index: int = 0) -> str:
    """Return the utterance stem for this node at the given hint level."""
    level = max(0, min(4, int(hint_level)))
    if level <= 0:
        return str(node["stem"])
    followups = list(node.get("followups") or [])
    probes = list(node.get("deep_probes") or [])
    if level == 1 and followups:
        idx = min(followup_index, len(followups) - 1)
        return followups[idx]
    if level == 2 and followups:
        idx = min(max(followup_index, 1), len(followups) - 1) if len(followups) > 1 else 0
        return followups[idx]
    if level >= 3 and probes:
        return probes[min(followup_index, len(probes) - 1)]
    if followups:
        return followups[0]
    return str(node["stem"])


def adjust_difficulty_ceiling(ceiling: int, score_0_100: float) -> int:
    c = max(1, min(5, int(ceiling)))
    if score_0_100 >= 78:
        return min(5, c + 1)
    if score_0_100 < 42:
        return max(1, c - 1)
    return c


def node_context_for_llm(node: QuestionNode | None, *, hint_level: int = 0) -> dict[str, Any] | None:
    if not node:
        return None
    return {
        "question_id": node["id"],
        "skill": _skill_key(node),
        "difficulty": node.get("difficulty"),
        "stem": node["stem"],
        "spoken_now": spoken_prompt(node, hint_level=hint_level),
        "rubric_strong": (node.get("rubric") or {}).get("strong"),
        "rubric_weak": (node.get("rubric") or {}).get("weak"),
        "hint_level": hint_level,
        "followups": node.get("followups") or [],
        "deep_probes": node.get("deep_probes") or [],
        "rule": (
            "Ask using spoken_now (or a tight paraphrase). Do not invent a new topic. "
            "Never reveal the answer. Probe; do not teach the solution. "
            "Hint ceiling is H3 — never H4 solution dump."
        ),
    }
