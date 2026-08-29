"""Realtime instruction builder (no network)."""

from app.realtime import interviewer_instructions, _turn_detection


def test_instructions_include_topics_and_duplex_role():
    text = interviewer_instructions(
        student_name="Priya Sharma",
        role_track="frontend",
        stage="qa",
        topics=["javascript", "dom"],
        briefing="Focus on accessibility.",
        include_coding=True,
        style="friendly",
        duration_minutes=17,
    )
    assert "Priya" in text
    assert "javascript" in text
    assert "accessibility" in text.lower() or "Focus on accessibility" in text
    assert "NexAI" in text
    assert "one" in text.lower()
    assert "Do NOT invent" not in text


def test_turn_detection_duplex():
    td = _turn_detection(create_response=True)
    assert td["create_response"] is True
    assert td["interrupt_response"] is True
