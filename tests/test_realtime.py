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
    assert "never invent another name" in text.lower()
    assert "javascript" in text
    assert "accessibility" in text.lower() or "Focus on accessibility" in text
    assert "NexAI" in text
    assert "CampusPay" in text
    assert "English" in text
    assert "invent" in text.lower()
    assert "Do NOT invent" not in text
    assert "GROUNDING" in text
    assert "MUST spend at least three questions" not in text


def test_instructions_agenda_order_survives_long_briefing():
    text = interviewer_instructions(
        student_name="Asha",
        role_track="sde_intern",
        stage="qa",
        topics=["Java/OOP", "DBMS", "OS"],
        briefing="Faculty wants OS last. " + ("pad " * 800),
        include_coding=True,
        style="friendly",
        duration_minutes=17,
    )
    assert "AGENDA" in text
    assert "item 1 (Java/OOP)" in text
    assert "1. Java/OOP" in text
    assert "2. DBMS" in text
    assert "3. OS" in text
    assert text.index("AGENDA") < text.index("FACULTY")
    assert "Do not skip ahead" in text


def test_turn_detection_server_vad():
    td = _turn_detection(create_response=False)
    assert td["type"] == "server_vad"
    assert td["silence_duration_ms"] == 700
    assert td["create_response"] is False
    assert td["interrupt_response"] is False
    semantic = _turn_detection(create_response=True, semantic=True)
    assert semantic["type"] == "semantic_vad"
    assert semantic["create_response"] is True
    assert semantic["interrupt_response"] is True
    live = _turn_detection(create_response=True)
    assert live["interrupt_response"] is True


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
    coding = coach_note(
        "Thanks — that wraps the technical questions. Let's move to coding.",
        topic="java",
    )
    assert "CODING ROUND" in coding
    assert "invent" not in coding.lower()
    assert "Stay on: java" not in coding


def test_coding_stage_instructions_stop_qa():
    text = interviewer_instructions(
        student_name="Priya Sharma",
        role_track="sde_intern",
        stage="idea",
        topics=["java", "dbms"],
    )
    assert "PROBLEM SOLVING" in text
    assert "technical Q&A round is OVER" in text
    assert "ArrayList" not in text


def test_scorer_prompt_is_compact():
    from app.llm import SCORER_SYSTEM, score_turn

    assert "JSON only" in SCORER_SYSTEM
    assert "do NOT write a question" in SCORER_SYSTEM
    assert len(SCORER_SYSTEM) < 1600
    assert callable(score_turn)
