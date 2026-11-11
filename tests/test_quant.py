from __future__ import annotations

import numpy as np
import pytest
import torch

from linguaprune.pruning import find_moe_blocks
from linguaprune.quant import (
    allocate_bits_by_coverage,
    allocate_like,
    expert_memory_bytes,
    fake_quantize,
    quantize_experts_,
    restore_experts,
    snapshot_experts,
)
from tests.olmoe_reference import TinyMoEModel


def make_model() -> TinyMoEModel:
    torch.manual_seed(0)
    return TinyMoEModel(num_layers=3, num_experts=8, top_k=2, hidden=32, intermediate=16).eval()


def test_error_shrinks_as_bits_grow_and_16_bits_is_lossless():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    errors = [(fake_quantize(w, b, 128) - w).abs().mean().item() for b in (2, 3, 4, 8)]
    assert errors == sorted(errors, reverse=True)
    torch.testing.assert_close(fake_quantize(w, 16), w)


def test_each_group_uses_at_most_2_pow_bits_levels():
    torch.manual_seed(0)
    q = fake_quantize(torch.randn(4, 256), bits=2, group_size=128)
    for group in q.reshape(-1, 128):
        assert group.unique().numel() <= 4


def test_fake_quantize_returns_new_tensor_and_validates():
    w = torch.randn(2, 128)
    original = w.clone()
    fake_quantize(w, 3)
    torch.testing.assert_close(w, original)
    with pytest.raises(ValueError):
        fake_quantize(torch.randn(2, 100), 4, group_size=128)
    with pytest.raises(ValueError):
        fake_quantize(w, 0)


def test_quantize_experts_touches_only_requested_experts_and_restore_undoes_it():
    model = make_model()
    snapshot = snapshot_experts(model)
    bits = np.full((3, 8), 16)
    bits[1, 5] = 2
    quantize_experts_(model, bits, group_size=16)
    blocks = find_moe_blocks(model)
    for layer, block in enumerate(blocks):
        for expert in range(8):
            changed = not torch.equal(block.experts.down_proj[expert], snapshot[layer][1][expert])
            assert changed == (layer == 1 and expert == 5)
    restore_experts(model, snapshot)
    for block, (gate_up, down) in zip(blocks, snapshot):
        assert torch.equal(block.experts.gate_up_proj, gate_up) and torch.equal(block.experts.down_proj, down)


def test_coverage_allocation_gives_most_used_experts_the_most_bits():
    score = np.array([[0.5, 0.2, 0.15, 0.1, 0.05]])
    # covered before each expert: 0, .5, .7, .85, .95 -> tiers by cuts (0.6, 0.9)
    np.testing.assert_array_equal(allocate_bits_by_coverage(score, (0.6, 0.9), (4, 3, 2)), [[4, 4, 3, 3, 2]])


def test_allocate_like_keeps_bit_counts_per_layer_but_follows_new_ranking():
    template = np.array([[4, 4, 3, 2], [4, 3, 3, 2]])
    other_score = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    bits = allocate_like(template, other_score)
    np.testing.assert_array_equal(np.sort(bits, axis=1), np.sort(template, axis=1))
    np.testing.assert_array_equal(bits[0], [2, 3, 4, 4])
    np.testing.assert_array_equal(bits[1], [4, 3, 3, 2])


def test_memory_accounts_for_bits_and_group_overhead():
    model = make_model()
    per_expert = 2 * 16 * 32 + 32 * 16  # gate_up [2I, H] + down [H, I]
    four_bit = expert_memory_bytes(model, np.full((3, 8), 4), group_size=16)
    assert four_bit == pytest.approx(3 * 8 * per_expert * (4 + 32 / 16) / 8)
    assert expert_memory_bytes(model, np.full((3, 8), 16)) == pytest.approx(3 * 8 * per_expert * 2)
    half_deleted = np.full((3, 8), 4)
    half_deleted[:, 4:] = 0
    assert expert_memory_bytes(model, half_deleted, group_size=16) == pytest.approx(four_bit / 2)
