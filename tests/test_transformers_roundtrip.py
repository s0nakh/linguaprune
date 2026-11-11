"""End-to-end check on the real transformers 5.x OLMoE classes (runs in CI; skipped if transformers<5).

A tiny random OLMoE is pruned, compared with the mask, saved with `save_pretrained` and loaded back
with `from_pretrained` — the exact path notebook 03 takes on the real model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("needs transformers 5.x (fused MoE expert layout)", allow_module_level=True)

from transformers import AutoModelForCausalLM, OlmoeConfig, OlmoeForCausalLM  # noqa: E402

from linguaprune import find_moe_blocks, masked_experts, prune_experts  # noqa: E402

LAYERS, EXPERTS, TOP_K, KEEP = 2, 8, 2, 5


def tiny_olmoe() -> OlmoeForCausalLM:
    torch.manual_seed(0)
    config = OlmoeConfig(vocab_size=128, hidden_size=32, intermediate_size=16, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=4, num_experts=EXPERTS,
                         num_experts_per_tok=TOP_K, max_position_embeddings=64, pad_token_id=0)
    model = OlmoeForCausalLM(config).eval()
    with torch.no_grad():  # random init so the router actually varies between experts
        for block in find_moe_blocks(model):
            block.gate.weight.normal_(std=1.0)
    return model


def kept_sets() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.sort(np.stack([rng.choice(EXPERTS, KEEP, replace=False) for _ in range(LAYERS)]), axis=1)


def test_real_olmoe_prune_matches_mask_and_survives_save_load(tmp_path):
    model = tiny_olmoe()
    kept = kept_sets()
    ids = torch.randint(1, 128, (1, 24))
    with torch.no_grad(), masked_experts(model, kept):
        masked = model(ids).logits

    report = prune_experts(model, kept)
    with torch.no_grad():
        pruned = model(ids).logits
    torch.testing.assert_close(pruned, masked, rtol=1e-5, atol=1e-5)
    assert report.experts_after == KEEP and model.config.num_experts == KEEP

    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert reloaded.config.num_experts == KEEP
    for block in find_moe_blocks(reloaded):
        assert block.gate.weight.shape[0] == KEEP
        assert block.experts.gate_up_proj.shape[0] == KEEP
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, pruned, rtol=1e-5, atol=1e-5)
