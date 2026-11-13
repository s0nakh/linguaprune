from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linguaprune.exam import ExamItem, check_against_reference, first_items_per_task, summarize
from linguaprune.llamacpp import (
    imatrix_command,
    log_softmax,
    merge_imatrix_command,
    quantize_command,
    score_items,
    span_logprob,
)


def test_log_softmax_rows_are_normalized_log_probabilities():
    rows = log_softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]]))
    assert np.allclose(np.exp(rows).sum(axis=-1), 1.0)
    assert np.isclose(rows[1, 0], np.log(0.5))


def test_span_logprob_uses_the_logits_one_position_earlier():
    logits = np.zeros((4, 5))
    logits[1, 3] = 10.0  # position 1 predicts token 3 at position 2
    ids = [0, 1, 3, 2]
    expected = log_softmax(logits[1:2])[0, 3] + log_softmax(logits[2:3])[0, 2]
    assert span_logprob(logits, ids, (2, 4)) == pytest.approx(expected)


def test_span_starting_at_the_first_token_is_rejected():
    with pytest.raises(ValueError):
        span_logprob(np.zeros((2, 3)), [0, 1], (0, 2))


class FakeLlama:
    """Logits favour token (position + 1) % vocab at every position."""

    def __init__(self, vocab: int):
        self.vocab, self.scores = vocab, None

    def reset(self):
        self.scores = None

    def eval(self, ids):
        self.scores = np.stack([np.eye(self.vocab)[(p + 1) % self.vocab] * 5 for p in range(len(ids))])


def test_score_items_writes_rows_in_notebook_07_format():
    items = [ExamItem((0, 1, 2, 3), (1, 4), (2, 3), {"task": "belebele", "lang": "kk", "label": 0}),
             ExamItem((0, 1), None, (1, 4), {"task": "kazmmlu", "lang": "ru", "label": 1})]
    rows = score_items(FakeLlama(vocab=6), items, [0, 1], "test")
    assert rows[0]["tokens"] == 3 and rows[0]["config"] == "test"
    assert rows[0]["choice"] == 0  # last position (3) favours token 4: neither 2 nor 3, both tie, argmax is the first
    assert rows[1]["logprob"] is None and rows[1]["choice"] == 0  # position 1 favours token 2... choices 1, 4 tie
    assert rows[0]["logprob"] > -1.0  # every scored token is the favoured one


def _scores(config: str, kk_logprob: float, mmlu_correct: bool) -> list[dict]:
    rows = []
    for lang in ("kk", "ru", "en"):
        rows.append({"config": config, "task": "belebele", "lang": lang, "label": 0,
                     "logprob": kk_logprob if lang == "kk" else -10.0, "tokens": 10, "choice": 0})
    for lang in ("kk", "ru"):
        rows.append({"config": config, "task": "kazmmlu", "lang": lang, "label": 1,
                     "logprob": None, "tokens": None, "choice": 1 if mmlu_correct else 0})
    for candidate, logprob in enumerate((-1.0, -2.0)):
        rows.append({"config": config, "task": "fact", "lang": "kk", "probe": 0, "candidate": candidate,
                     "logprob": logprob, "tokens": 1, "choice": None})
    return rows


def test_summarize_reports_growth_against_the_baseline():
    scores = pd.DataFrame(_scores("original", -10.0, True) + _scores("small", -11.0, False))
    table = summarize(scores, {"original": 60.0, "small": 15.0}, baseline="original")
    assert table.loc["small", "growth_kk"] == pytest.approx(np.exp(0.1) - 1)
    assert table.loc["small", "growth_ru"] == pytest.approx(0.0)
    assert table.loc["small", "compression"] == pytest.approx(4.0)
    assert table.loc["original", "kazmmlu_kk"] == 1.0 and table.loc["small", "kazmmlu_kk"] == 0.0
    assert table.loc["small", "facts"] == 1.0
    assert list(table.index) == ["original", "small"]


def test_summarize_requires_the_baseline():
    with pytest.raises(KeyError):
        summarize(pd.DataFrame(_scores("small", -11.0, False)), {"small": 15.0}, baseline="original")


def test_reference_check_catches_misaligned_rows():
    items = [ExamItem((0, 1, 2), (1, 3), None, {"task": "fact", "lang": "kk"})]
    good = pd.DataFrame({"task": ["fact"], "lang": ["kk"], "tokens": [2]})
    check_against_reference(items, good, n_ctx=8)
    with pytest.raises(RuntimeError):
        check_against_reference(items, good.assign(tokens=[3]), n_ctx=8)
    with pytest.raises(RuntimeError):
        check_against_reference(items, good.assign(lang=["ru"]), n_ctx=8)
    with pytest.raises(RuntimeError):
        check_against_reference(items, good, n_ctx=2)


def test_first_items_per_task_keeps_exam_order():
    metas = [("a", "kk"), ("a", "kk"), ("b", "kk"), ("a", "kk"), ("b", "kk"), ("b", "kk")]
    items = [ExamItem((0,), None, None, {"task": t, "lang": l}) for t, l in metas]
    assert first_items_per_task(items, 2) == [0, 1, 2, 4]


def test_tool_command_lines():
    tool = Path("bin/llama-imatrix")
    cmd = imatrix_command(tool, Path("m.gguf"), Path("t.txt"), Path("o.gguf"), chunks=3)
    assert cmd[:2] == ["bin/llama-imatrix", "-m"] and cmd[-2:] == ["--chunks", "3"] and "--fit" in cmd
    assert merge_imatrix_command(tool, [Path("a"), Path("b")], Path("c")) == \
        ["bin/llama-imatrix", "--in-file", "a", "--in-file", "b", "-o", "c"]
    with pytest.raises(ValueError):
        merge_imatrix_command(tool, [Path("a")], Path("c"))
    q = quantize_command(Path("q"), Path("src.gguf"), Path("out.gguf"), "IQ2_M", Path("im.gguf"), ["attn_v=q8_0"])
    assert q[:4] == ["q", "--allow-requantize", "--imatrix", "im.gguf"]
    assert q[4:6] == ["--tensor-type", "attn_v=q8_0"] and q[6:9] == ["src.gguf", "out.gguf", "IQ2_M"]
    with pytest.raises(ValueError):
        quantize_command(Path("q"), Path("s"), Path("o"), "IQ2_M", None, ["attn_v"])
