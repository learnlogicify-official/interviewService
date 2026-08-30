"""Duplex utterance buffer — no sticky IDK prefixes."""

from app import duplex_score


def test_stale_idk_chip_does_not_prefix_next_answer():
    state: dict = {}
    assert duplex_score.take_scoreable_text(state, "I couldn't get. I don't know.", stage="qa") is None
    assert state.get("utterance_buffer")

    # Force the chip to look stale (older than 12s).
    state["utterance_buffer_at"] = 0
    out = duplex_score.take_scoreable_text(
        state,
        "I used Ruby on Rails and GitHub to track changes and collaborate with the team.",
        stage="qa",
    )
    assert out is not None
    assert "don't know" not in out.lower()
    assert "Ruby on Rails" in out
    assert not state.get("utterance_buffer")


def test_fresh_scoreable_answer_replaces_unrelated_chip():
    state: dict = {}
    duplex_score.append_utterance_buffer(state, "I couldn't get.")
    out = duplex_score.take_scoreable_text(
        state,
        "Primary keys guarantee uniqueness and not null; a duplicate insert is rejected.",
        stage="qa",
    )
    assert out is not None
    assert "couldn't get" not in out.lower()
    assert "Primary keys" in out


def test_continuation_still_merges():
    state: dict = {}
    assert duplex_score.take_scoreable_text(state, "I will use a hash map", stage="qa") is None
    out = duplex_score.take_scoreable_text(
        state,
        "I will use a hash map to count frequencies and then report keys with count greater than one.",
        stage="qa",
    )
    assert out is not None
    assert "hash map" in out.lower()
    assert "frequencies" in out.lower()
