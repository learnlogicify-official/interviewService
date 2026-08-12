"""Lightweight communication / voice transcript metrics (text-side)."""

from __future__ import annotations

import re
from typing import Any


FILLERS = re.compile(
    r"\b(um+|uh+|er+|ah+|like|you know|basically|actually|sort of|kind of|i mean|right)\b",
    re.I,
)
SENTENCE_SPLIT = re.compile(r"[.!?]+")


def analyze_utterance(text: str, *, duration_sec: float | None = None) -> dict[str, Any]:
    """
    Derive communication metrics from transcript text.
    Optional duration_sec (from client VAD) improves WPM accuracy.
    """
    clean = " ".join((text or "").split()).strip()
    words = re.findall(r"[A-Za-z0-9_']+", clean)
    word_count = len(words)
    fillers = FILLERS.findall(clean)
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(clean) if s.strip()]
    avg_sentence = (word_count / max(1, len(sentences))) if sentences else float(word_count)

    # Rough pause proxy: long comma/ellipsis clusters and repeated spaces already collapsed.
    long_pause_markers = len(re.findall(r"\.\.\.|—|–", text or ""))

    if duration_sec and duration_sec > 0.4:
        wpm = (word_count / duration_sec) * 60.0
    else:
        # Assume ~150 wpm speaking rate if timing unknown — used only as soft prior.
        wpm = 150.0 if word_count else 0.0

    # Structure: presence of reasoning cues.
    structure_hits = sum(
        1
        for cue in ("because", "first", "second", "for example", "trade-off", "tradeoff", "so ")
        if cue in clean.lower()
    )
    structure = min(100.0, 40 + structure_hits * 15 + (10 if word_count >= 40 else 0))

    clarity = 100.0
    clarity -= min(40.0, len(fillers) * 4.5)
    if word_count < 12:
        clarity -= 20
    if avg_sentence > 40:
        clarity -= 10
    clarity = max(0.0, min(100.0, clarity))

    relevance_proxy = min(100.0, 50 + min(40, word_count) + structure_hits * 5)

    return {
        "word_count": word_count,
        "filler_count": len(fillers),
        "fillers": [f.lower() for f in fillers[:12]],
        "sentence_count": len(sentences),
        "avg_sentence_words": round(avg_sentence, 1),
        "long_pause_markers": long_pause_markers,
        "speaking_rate_wpm": round(wpm, 1),
        "clarity": round(clarity, 1),
        "structure": round(structure, 1),
        "relevance_proxy": round(relevance_proxy, 1),
        "duration_sec": round(float(duration_sec or 0), 2),
    }


def blend_communication(prev: float, metrics: dict[str, Any]) -> float:
    """Blend utterance metrics into running communication score 0–100."""
    sample = (
        float(metrics.get("clarity", 50)) * 0.45
        + float(metrics.get("structure", 50)) * 0.35
        + float(metrics.get("relevance_proxy", 50)) * 0.2
    )
    if prev <= 0:
        return round(sample, 1)
    return round(prev * 0.7 + sample * 0.3, 1)
