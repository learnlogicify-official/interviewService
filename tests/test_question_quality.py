"""Question-quality gates — no network."""

from app.llm import (
    default_ack,
    is_no_knowledge_answer,
    is_off_lock_dsa,
    is_unclear_question_response,
    is_vague_question,
    strip_spoken_meta,
)


def test_unclear_does_not_score_as_no_knowledge():
    samples = [
        "I really didn't get your question properly.",
        "No, I don't know. I don't understand really by how you do use Java in a small project. "
        "That question itself is absurd.",
        "what this meant by how Q is implemented using you know like any programming language",
    ]
    for text in samples:
        assert is_unclear_question_response(text), text
        assert not is_no_knowledge_answer(text), text


def test_plain_idk_is_still_no_knowledge():
    assert is_no_knowledge_answer("I don't know")
    assert not is_unclear_question_response("I don't know")


def test_vague_shapes_from_real_session():
    bads = [
        "Thanks — let's move to the next topic. Let's start with java and programming fundamentals. "
        "Walk me through how you'd approach it in a real project, and call out one trade-off you'd have to make.",
        "How would you use Java in a small project? Any trade-offs you faced?",
        "Can you explain the difference between a stack and a queue, and provide a real-world application for each?",
        "Can you explain how a queue is implemented in programming?",
        "Explain Big-O for time and space. Give an O(n) time and O(1) extra space example.",
    ]
    for text in bads:
        assert is_vague_question(text) or is_off_lock_dsa(
            text, ["java", "programming fundamentals"]
        ), text


def test_concrete_java_question_is_allowed():
    good = (
        "In Java, you loop over an ArrayList of orders and call remove() on the cancelled ones "
        "inside the loop. What happens at runtime, and what would you do instead?"
    )
    assert not is_vague_question(good)
    assert not is_off_lock_dsa(good, ["java", "programming fundamentals"])


def test_strip_rude_and_scores():
    raw = "That's okay — we'll mark this topic as weak and move on. Explain Big-O for time and space."
    cleaned = strip_spoken_meta(raw)
    assert "weak" not in cleaned.lower()
    assert "100" not in strip_spoken_meta("Thanks — noted (40/100). Next question?")


def test_friendly_ack():
    assert default_ack("friendly") == "Thanks —"
    assert default_ack("strict") == "Got it."
