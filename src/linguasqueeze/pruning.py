"""Choose experts per layer and remove the rest from a Mixture-of-Experts model.

Supports the transformers 5.x MoE layout used by OLMoE (and Qwen2-MoE-style blocks):
each MoE block has a `gate` router (`weight` of shape [experts, hidden], `top_k`,
`norm_topk_prob`) and an `experts` module with fused 3D parameters
`gate_up_proj` [experts, 2*intermediate, hidden] and `down_proj` [experts, hidden, intermediate].
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MoEBlock:
    """One MoE layer: its router and its expert weights."""

    name: str
    gate: nn.Module
    experts: nn.Module


@dataclass(frozen=True)
class PruneReport:
    experts_before: int
    experts_after: int
    params_before: int
    params_after: int

    @property
    def removed_fraction(self) -> float:
        return 1 - self.params_after / self.params_before


def find_moe_blocks(model: nn.Module) -> list[MoEBlock]:
    """Find every MoE block, in layer order. Fails loudly on an unsupported layout."""
    blocks = [
        MoEBlock(name, module.gate, module.experts)
        for name, module in model.named_modules()
        if hasattr(module, "gate") and hasattr(getattr(module, "experts", None), "gate_up_proj")
    ]
    if not blocks:
        raise ValueError("No MoE blocks with `gate` and fused `experts.gate_up_proj` found (transformers 5.x layout).")
    for block in blocks:
        missing = [a for a in ("weight", "top_k", "norm_topk_prob") if not hasattr(block.gate, a)]
        if missing or not hasattr(block.experts, "down_proj"):
            raise ValueError(f"Unsupported MoE block {block.name}: missing {missing or ['experts.down_proj']}")
    return blocks


def top_experts(score: np.ndarray, keep: int) -> np.ndarray:
    """Indices of the `keep` highest-scoring experts in every layer, sorted. score: [layers, experts]."""
    if not 0 < keep <= score.shape[1]:
        raise ValueError(f"keep must be in 1..{score.shape[1]}, got {keep}")
    return np.sort(np.argsort(-score, axis=1, kind="stable")[:, :keep], axis=1)


def row_share(counts: np.ndarray) -> np.ndarray:
    """Normalize per-layer counts so every row sums to 1."""
    return counts / counts.sum(axis=1, keepdims=True)


def language_frequency_score(counts_by_language: dict[str, np.ndarray], languages: Sequence[str]) -> np.ndarray:
    """Mean per-layer selection share over the target languages (equal weight per language)."""
    return np.mean([row_share(counts_by_language[lang]) for lang in languages], axis=0)


def _validate_kept(blocks: list[MoEBlock], kept: np.ndarray) -> int:
    kept = np.asarray(kept)
    if kept.ndim != 2 or kept.shape[0] != len(blocks):
        raise ValueError(f"kept must have shape [layers={len(blocks)}, K], got {kept.shape}")
    num_experts = blocks[0].gate.weight.shape[0]
    keep = kept.shape[1]
    top_k = blocks[0].gate.top_k
    if keep < top_k:
        raise ValueError(f"Cannot keep {keep} experts per layer: the router activates {top_k}.")
    for layer, row in enumerate(kept):
        if len(set(row.tolist())) != keep or row.min() < 0 or row.max() >= num_experts:
            raise ValueError(f"Layer {layer}: kept indices must be {keep} distinct values in 0..{num_experts - 1}")
    return keep


def masked_routing(router: nn.Module, logits: torch.Tensor, mask: torch.Tensor):
    """The OLMoE router computation (transformers 5.x) with removed experts masked to -inf."""
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dtype=torch.float, dim=-1)
    top_value, top_index = torch.topk(probs, router.top_k, dim=-1)
    if router.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return masked_logits, top_value.to(logits.dtype), top_index


def build_mask(kept_layer: Sequence[int], router: nn.Module) -> torch.Tensor:
    """0 for kept experts, -inf for removed ones, on the router's device and dtype."""
    num_experts = router.weight.shape[0]
    mask = torch.full((num_experts,), float("-inf"), device=router.weight.device)
    mask[torch.as_tensor(np.asarray(kept_layer), device=router.weight.device)] = 0.0
    return mask.to(router.weight.dtype)


@contextmanager
def masked_experts(model: nn.Module, kept: np.ndarray) -> Iterator[None]:
    """Temporarily simulate removing experts: the router can no longer pick them. Nothing is deleted."""
    blocks = find_moe_blocks(model)
    _validate_kept(blocks, kept)
    handles = []
    for block, row in zip(blocks, np.asarray(kept)):
        mask = build_mask(row, block.gate)

        def hook(router, _inputs, output, mask=mask):
            if not (isinstance(output, tuple) and len(output) == 3):
                raise RuntimeError("Router did not return (logits, scores, indices): unsupported transformers version.")
            return masked_routing(router, output[0], mask)

        handles.append(block.gate.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _keep_rows(param: nn.Parameter, index_cpu: torch.Tensor) -> nn.Parameter:
    index = index_cpu.to(param.device)
    return nn.Parameter(param.data.index_select(0, index).clone(), requires_grad=False)


@torch.no_grad()
def prune_experts(model: nn.Module, kept: np.ndarray) -> PruneReport:
    """Physically delete every expert not listed in `kept` ([layers, K]) and update the config.

    Modifies `model` in place: copying a multi-gigabyte model just to return a new one is not
    practical. After this call `model.save_pretrained(...)` writes a smaller checkpoint that
    loads with the standard `from_pretrained`.
    """
    blocks = find_moe_blocks(model)
    keep = _validate_kept(blocks, kept)
    experts_before = blocks[0].gate.weight.shape[0]
    params_before = count_parameters(model)
    for block, row in zip(blocks, np.asarray(kept)):
        index = torch.as_tensor(row, dtype=torch.long)
        block.gate.weight = _keep_rows(block.gate.weight, index)
        block.experts.gate_up_proj = _keep_rows(block.experts.gate_up_proj, index)
        block.experts.down_proj = _keep_rows(block.experts.down_proj, index)
        for module in (block.gate, block.experts):
            if hasattr(module, "num_experts"):
                module.num_experts = keep
    for holder in (model, getattr(model, "config", None)):
        if holder is not None and hasattr(holder, "num_experts"):
            holder.num_experts = keep
    return PruneReport(experts_before, keep, params_before, count_parameters(model))
