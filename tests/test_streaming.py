"""Layer-by-layer streaming must give the same scores as the model's normal forward (real transformers 5.x)."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("needs transformers 5.x", allow_module_level=True)

from transformers import AutoModelForCausalLM, OlmoeConfig, Qwen3MoeConfig  # noqa: E402

from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate  # noqa: E402

TINY = dict(vocab_size=128, hidden_size=32, intermediate_size=16, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, num_experts=8, num_experts_per_tok=2, max_position_embeddings=64, pad_token_id=0)


def tiny_config(kind: str):
    if kind == "olmoe":
        return OlmoeConfig(**TINY)
    return Qwen3MoeConfig(**TINY, moe_intermediate_size=16, head_dim=8, norm_topk_prob=True, decoder_sparse_step=1)


def reference_scores(model, seq: ScoredSequence):
    with torch.no_grad():
        logits = model(torch.tensor([seq.ids])).logits[0].float()
    start, end = seq.span
    logprob = torch.log_softmax(logits[start - 1:end - 1], -1).gather(1, torch.tensor(seq.ids[start:end])[:, None]).sum()
    return logprob.item(), logits[-1, list(seq.choice_ids)].tolist()


@pytest.mark.parametrize("kind", ["olmoe", "qwen3_moe"])
def test_streamed_scores_match_normal_forward(tmp_path, kind):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tiny_config(kind), dtype=torch.float32).eval()
    model.save_pretrained(tmp_path)
    gen = torch.Generator().manual_seed(1)
    sequences = [ScoredSequence(ids=tuple(torch.randint(1, 128, (n,), generator=gen).tolist()), span=(2, n),
                                choice_ids=(5, 6, 7)) for n in (7, 12, 20, 9)]  # different lengths -> padding
    streamed = StreamedModel(tmp_path, device="cpu", dtype=torch.float32)
    calls = []
    results = stream_evaluate(streamed, sequences, configs=["original"], max_tokens=40,
                              prepare=lambda cfg, i, layer, inputs: calls.append((cfg, i)), log=lambda _: None)
    assert calls == [("original", i) for i in range(3)]
    for seq, got in zip(sequences, results["original"]):
        want_logprob, want_choices = reference_scores(model, seq)
        assert got.span_tokens == len(seq.ids) - 2
        assert got.span_logprob == pytest.approx(want_logprob, abs=1e-4)
        assert got.choice_logits == pytest.approx(want_choices, abs=1e-4)


def test_prepare_changes_only_its_configuration_exactly(tmp_path):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(tiny_config("olmoe"), dtype=torch.float32).eval()
    model.save_pretrained(tmp_path)
    seq = ScoredSequence(ids=tuple(range(1, 11)), span=(1, 10), choice_ids=(3, 4))

    def scale_experts(cfg, index, layer, inputs):
        if cfg == "scaled":
            with torch.no_grad():
                layer.mlp.experts.down_proj.mul_(200.0)

    streamed = StreamedModel(tmp_path, device="cpu", dtype=torch.float32)
    out = stream_evaluate(streamed, [seq], ["original", "scaled"], prepare=scale_experts, log=lambda _: None)

    original_logprob, original_choices = reference_scores(model, seq)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.experts.down_proj.mul_(200.0)
    scaled_logprob, scaled_choices = reference_scores(model, seq)

    assert out["original"][0].span_logprob == pytest.approx(original_logprob, abs=1e-4)
    assert out["original"][0].choice_logits == pytest.approx(original_choices, abs=1e-4)
    assert out["scaled"][0].span_logprob == pytest.approx(scaled_logprob, abs=1e-4)
    assert out["scaled"][0].choice_logits == pytest.approx(scaled_choices, abs=1e-4)
