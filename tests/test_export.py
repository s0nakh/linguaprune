"""Real low-bit checkpoints: standard GPTQ layout, exact round trip, and identical scores from the file."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from linguaprune.export import from_gptq, gptq_bits, pack_rows, to_gptq, unpack_rows
from linguaprune.gptq import gptq_quantize_batched, rtn_quantize


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_unpack_round_trip(bits):
    g = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 2**bits, (96, 5), generator=g)
    packed = pack_rows(codes, bits)
    assert packed.dtype == torch.int32 and packed.shape == (96 * bits // 32, 5)
    assert torch.equal(unpack_rows(packed, bits), codes)


@pytest.mark.parametrize("bits", [3, 4])
def test_packing_is_a_little_endian_bit_stream(bits):
    """Value i of a column occupies bits [i*bits, (i+1)*bits) of the column's concatenated 32-bit words."""
    codes = torch.randint(0, 2**bits, (64, 3), generator=torch.Generator().manual_seed(0))
    packed = pack_rows(codes, bits)
    for col in range(3):
        stream = sum((int(w) & 0xFFFFFFFF) << (32 * k) for k, w in enumerate(packed[:, col]))
        for i in range(64):
            assert (stream >> (i * bits)) & (2**bits - 1) == codes[i, col]


def test_4bit_layout_matches_the_autogptq_reference_dequantizer():
    """The unpacking used by AutoGPTQ / GPTQModel's torch kernel for 4-bit v1 checkpoints."""
    torch.manual_seed(0)
    weight = torch.randn(64, 256)
    q, scale, zero = rtn_quantize(weight, 4, group_size=128)
    t = to_gptq(q, scale, zero, bits=4, group_size=128)
    wf = torch.arange(0, 32, 4, dtype=torch.int32)
    w = torch.bitwise_and(torch.bitwise_right_shift(t["qweight"].unsqueeze(1), wf.view(1, -1, 1)), 15)
    w = w.reshape(-1, t["qweight"].shape[1])  # [in, out]
    z = torch.bitwise_and(torch.bitwise_right_shift(t["qzeros"].unsqueeze(2), wf.view(1, 1, -1)), 15)
    z = z.reshape(t["scales"].shape) + 1  # v1 format stores zero - 1
    reference = (t["scales"].float()[t["g_idx"].long()] * (w - z[t["g_idx"].long()]).float()).T
    assert torch.equal(reference, q)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_weights_survive_the_checkpoint_bit_exactly(bits):
    torch.manual_seed(bits)
    weight = torch.randn(96, 256) * 0.02
    weight[:, :128] = weight[:, :128].abs()  # an all-positive group exercises the zero-point clamp
    q, scale, zero = rtn_quantize(weight, bits, group_size=128)
    tensors = to_gptq(q, scale, zero, bits, group_size=128)
    assert gptq_bits(tensors) == bits
    assert tensors["scales"].dtype == torch.float16 and tensors["g_idx"].dtype == torch.int32
    assert torch.equal(from_gptq(tensors), q)


def test_gptq_weights_survive_the_checkpoint_bit_exactly():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(3, 32, 256, generator=g)
    X = torch.randn(512, 256, generator=g) @ torch.randn(256, 256, generator=g) / 16
    H = (X.T @ X).expand(3, 256, 256).clone()
    Q, scales, zeros = gptq_quantize_batched(W, H, bits=3, group_size=128, return_params=True)
    assert scales.shape == zeros.shape == (3, 32, 2)
    for b in range(3):
        assert torch.equal(from_gptq(to_gptq(Q[b], scales[b], zeros[b], 3, 128)), Q[b])


def test_off_grid_weights_are_refused():
    q, scale, zero = rtn_quantize(torch.randn(8, 128), 4, group_size=128)
    with pytest.raises(ValueError):
        to_gptq(q + 1e-3, scale, zero, 4, 128)


# ---- end to end on a tiny real transformers-5 Qwen3-MoE ---------------------------------------------------

def _tiny_qwen3(tmp_path):
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("needs transformers 5.x")
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    config = Qwen3MoeConfig(vocab_size=128, hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            num_experts=8, num_experts_per_tok=2, max_position_embeddings=64, pad_token_id=0,
                            norm_topk_prob=True, decoder_sparse_step=1)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
    model.save_pretrained(tmp_path / "original")
    return tmp_path / "original"


def test_written_checkpoint_reproduces_the_simulated_scores_exactly(tmp_path):
    source = _tiny_qwen3(tmp_path)
    from linguaprune.export import GPTQCheckpointWriter
    from linguaprune.gptq import make_quantization_prepare
    from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate

    gen = torch.Generator().manual_seed(1)
    exam = [ScoredSequence(ids=tuple(torch.randint(1, 128, (n,), generator=gen).tolist()), span=(2, n),
                           choice_ids=(5, 6, 7)) for n in (9, 14, 21)]
    calibration = [ScoredSequence(ids=tuple(torch.randint(1, 128, (32,), generator=gen).tolist())) for _ in range(6)]
    bits = np.array([[4, 3, 4, 3, 4, 4, 3, 4], [3, 4, 4, 4, 3, 4, 4, 4]])

    streamed = StreamedModel(source, device="cpu", dtype=torch.float32)
    writer = GPTQCheckpointWriter(tmp_path / "packed", streamed.reader, group_size=32, default_bits=4,
                                  attention_bits=8, description={"name": "tiny"})
    prepare, stats = make_quantization_prepare(
        streamed, {"q": {"method": "gptq", "bits": bits}}, group_size=32, attention_bits=8,
        log=lambda _: None, on_quantized=lambda c, i, layer, rec: writer.write_layer(c, i, layer, rec))
    simulated = stream_evaluate(streamed, exam, ["q"], prepare=prepare, calibration={"q": calibration},
                                log=lambda _: None)["q"]
    summary = writer.finish(num_layers=2)

    packed_dir = tmp_path / "packed"
    weight_map = json.loads((packed_dir / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.mlp.experts.1.gate_proj.qweight" in weight_map
    assert "model.layers.0.self_attn.q_proj.qweight" in weight_map
    assert not any(k.endswith("experts.0.down_proj.weight") for k in weight_map)
    config = json.loads((packed_dir / "config.json").read_text())["quantization_config"]
    assert config["quant_method"] == "gptq" and config["bits"] == 4
    assert {"bits": 3} in config["dynamic"].values() and {"bits": 8} in config["dynamic"].values()
    assert summary["expert_bits_per_layer"] == bits.tolist()

    from_file = stream_evaluate(StreamedModel(packed_dir, device="cpu", dtype=torch.float32), exam, ["file"],
                                log=lambda _: None)["file"]
    for a, b in zip(simulated, from_file):
        assert a.span_logprob == b.span_logprob
        assert a.choice_logits == b.choice_logits
