from __future__ import annotations

import numpy as np
import pytest
import torch

from linguaprune import (
    count_parameters,
    find_moe_blocks,
    language_frequency_score,
    masked_experts,
    prune_experts,
    top_experts,
)
from tests.olmoe_reference import TinyMoEModel

LAYERS, EXPERTS, TOP_K, KEEP = 3, 16, 4, 8


def make_model(norm_topk_prob: bool = False, seed: int = 0) -> TinyMoEModel:
    torch.manual_seed(seed)
    return TinyMoEModel(num_layers=LAYERS, num_experts=EXPERTS, top_k=TOP_K, norm_topk_prob=norm_topk_prob).eval()


def random_kept(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(np.stack([rng.choice(EXPERTS, KEEP, replace=False) for _ in range(LAYERS)]), axis=1)


@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_physical_removal_matches_masked_routing(norm_topk_prob):
    model = make_model(norm_topk_prob)
    kept = random_kept()
    inputs = torch.randn(2, 64, 32)
    with torch.no_grad(), masked_experts(model, kept):
        masked = model(inputs)

    prune_experts(model, kept)
    with torch.no_grad():
        pruned = model(inputs)

    torch.testing.assert_close(pruned, masked, rtol=1e-5, atol=1e-5)


def test_masking_changes_outputs_and_is_undone_after_context():
    model = make_model()
    inputs = torch.randn(1, 32, 32)
    with torch.no_grad():
        original = model(inputs)
        with masked_experts(model, random_kept()):
            masked = model(inputs)
        restored = model(inputs)
    assert not torch.allclose(masked, original)
    torch.testing.assert_close(restored, original)


def test_prune_updates_shapes_counts_and_config():
    model = make_model()
    before = count_parameters(model)
    report = prune_experts(model, random_kept())

    for block in find_moe_blocks(model):
        assert block.gate.weight.shape[0] == KEEP
        assert block.experts.gate_up_proj.shape[0] == KEEP
        assert block.experts.down_proj.shape[0] == KEEP
        assert block.gate.num_experts == KEEP and block.experts.num_experts == KEEP
    assert model.config.num_experts == KEEP and model.num_experts == KEEP
    assert report.experts_before == EXPERTS and report.experts_after == KEEP
    assert report.params_before == before and report.params_after == count_parameters(model) < before


def test_pruned_weights_are_the_kept_rows():
    model = make_model()
    kept = random_kept()
    original_rows = [block.experts.down_proj.detach()[kept[i]].clone() for i, block in enumerate(find_moe_blocks(model))]
    prune_experts(model, kept)
    for rows, block in zip(original_rows, find_moe_blocks(model)):
        torch.testing.assert_close(block.experts.down_proj.detach(), rows)


@pytest.mark.parametrize(
    "kept, message",
    [
        (np.tile(np.arange(TOP_K - 1), (LAYERS, 1)), "router activates"),
        (np.tile(np.arange(KEEP), (LAYERS - 1, 1)), "shape"),
        (np.tile(np.r_[0, 0, np.arange(2, KEEP)], (LAYERS, 1)), "distinct"),
        (np.tile(np.r_[np.arange(KEEP - 1), EXPERTS], (LAYERS, 1)), "distinct"),
    ],
)
def test_invalid_kept_sets_are_rejected(kept, message):
    with pytest.raises(ValueError, match=message):
        prune_experts(make_model(), kept)


def test_non_moe_model_is_rejected():
    with pytest.raises(ValueError, match="No MoE blocks"):
        find_moe_blocks(torch.nn.Linear(4, 4))


def test_top_experts_picks_highest_scores_per_layer():
    score = np.array([[0.1, 0.9, 0.5, 0.3], [0.8, 0.1, 0.2, 0.7]])
    np.testing.assert_array_equal(top_experts(score, 2), [[1, 2], [0, 3]])
    with pytest.raises(ValueError):
        top_experts(score, 0)


def test_language_frequency_score_weights_languages_equally():
    counts = {"kk": np.array([[3, 1]]), "ru": np.array([[100, 300]])}
    np.testing.assert_allclose(language_frequency_score(counts, ["kk", "ru"]), [[0.5, 0.5]])
