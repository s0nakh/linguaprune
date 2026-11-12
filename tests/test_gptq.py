from __future__ import annotations

import numpy as np
import pytest
import torch

from linguaprune.gptq import gptq_quantize, quantize_layer_gptq_
from linguaprune.quant import fake_quantize


def correlated_inputs(n: int, dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(dim, dim, generator=g) / dim**0.5
    return torch.randn(n, dim, generator=g) @ mix + 0.1 * torch.randn(n, dim, generator=g)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_gptq_has_lower_output_error_than_rtn_on_calibration_data(bits):
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    X = correlated_inputs(2048, 256)
    H = X.T @ X
    err = lambda Q: ((X @ W.T - X @ Q.T) ** 2).mean().item()
    assert err(gptq_quantize(W, H, bits, group_size=128)) < err(fake_quantize(W, bits, 128))


def test_gptq_uses_at_most_2_pow_bits_levels_per_row_and_group():
    torch.manual_seed(0)
    W = torch.randn(8, 256)
    X = correlated_inputs(512, 256)
    Q = gptq_quantize(W, X.T @ X, bits=2, group_size=128)
    for row in Q:
        for group in row.reshape(-1, 128):
            assert group.unique().numel() <= 4


def test_gptq_16_bits_is_identity_and_shapes_are_validated():
    W = torch.randn(4, 128)
    torch.testing.assert_close(gptq_quantize(W, torch.eye(128), 16), W)
    with pytest.raises(ValueError):
        gptq_quantize(W, torch.eye(64), 4)


def test_dead_input_columns_do_not_break_gptq():
    torch.manual_seed(0)
    W = torch.randn(4, 128)
    X = torch.randn(256, 128)
    X[:, :10] = 0
    Q = gptq_quantize(W, X.T @ X, 4)
    assert torch.isfinite(Q).all()


def test_layer_quantization_on_reference_block_reduces_error_vs_rtn():
    from tests.olmoe_reference import TinyMoEModel

    torch.manual_seed(0)
    model = TinyMoEModel(num_layers=1, num_experts=4, top_k=2, hidden=128, intermediate=128).eval()
    block = model.layers[0]["mlp"]
    layer = torch.nn.Module()
    layer.mlp = block
    x = correlated_inputs(1024, 128).reshape(8, 128, 128)
    mask = torch.ones(8, 128, dtype=torch.long)
    with torch.no_grad():
        reference = block(x)

    def inputs():
        yield x, mask

    def run_layer(hidden, _mask):
        return layer.mlp(hidden)

    snapshot = [p.detach().clone() for p in (block.experts.gate_up_proj, block.experts.down_proj)]
    stats = quantize_layer_gptq_(layer, np.full(4, 3), inputs, run_layer, group_size=128, attention_bits=None)
    with torch.no_grad():
        gptq_err = ((block(x) - reference) ** 2).mean().item()
        block.experts.gate_up_proj.copy_(fake_quantize(snapshot[0], 3, 128))
        block.experts.down_proj.copy_(fake_quantize(snapshot[1], 3, 128))
        rtn_err = ((block(x) - reference) ** 2).mean().item()
    assert sum(stats["tokens_per_expert"]) == 1024 * 2  # every real token reaches top_k=2 experts
    assert gptq_err < rtn_err
