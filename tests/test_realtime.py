"""Realtime instruction builder (no network)."""

from app.realtime import interviewer_instructions, _turn_detection, coach_note


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
        resume_dossier={"projects": [{"name": "CampusPay"}], "skills": ["React"]},
    )
    assert "Priya" in text
    assert "javascript" in text
    assert "accessibility" in text.lower() or "Focus on accessibility" in text
    assert "NexAI" in text
    assert "CampusPay" in text
    assert "invent" in text.lower()
    assert "Do NOT invent" not in text


def test_turn_detection_server_vad():
    td = _turn_detection(create_response=False)
    assert td["type"] == "server_vad"
    assert td["silence_duration_ms"] == 700
    assert td["create_response"] is False
    assert td["interrupt_response"] is True
    semantic = _turn_detection(create_response=True, semantic=True)
    assert semantic["type"] == "semantic_vad"
    assert semantic["create_response"] is True


def test_coach_note_is_not_a_script():
    note = coach_note(
        "Thanks — in Java you loop an ArrayList and call remove. What happens?",
        topic="java",
    )
    assert "invent" in note.lower()
    assert "ArrayList" not in note
    assert "Stay on: java" in note
    wrap = coach_note("Thanks for your time today.", wrap=True)
    assert wrap.startswith("WRAP:")
