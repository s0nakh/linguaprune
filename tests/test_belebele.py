from linguaprune.belebele import token_span


def test_token_span_covers_passage_tokens_only():
    # "Passage: Hello world\nQ" -> tokens: "Passage", ":", " Hello", " world", "\n", "Q"
    offsets = [(0, 7), (7, 8), (8, 14), (14, 20), (20, 21), (21, 22)]
    passage_start, passage_end = len("Passage: "), len("Passage: Hello world")
    assert token_span(offsets, passage_start, passage_end) == (2, 4)


def test_token_span_runs_to_end_when_passage_is_last():
    offsets = [(0, 7), (7, 8), (8, 14)]
    assert token_span(offsets, 9, 14) == (2, 3)
