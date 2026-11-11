"""Language-aware mixed-precision quantization of MoE experts (simulated, for quality measurement).

Instead of deleting rarely used experts (which deletes their knowledge), every expert is kept and
gets a bit-width: experts the target languages use a lot stay precise, rarely used ones are
compressed hard. Quantization here is *simulated* ("fake quant"): weights are rounded to the
quantization grid and stored back in fp16, so quality can be measured with the normal model.
The memory a real low-bit format would need is computed analytically by `expert_memory_bytes`.

Scheme: asymmetric round-to-nearest (RTN), one fp16 scale and one fp16 zero point per group of
`group_size` consecutive input weights.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from linguaprune.pruning import find_moe_blocks

NO_QUANT_BITS = 16
SCALE_ZERO_BITS = 32  # fp16 scale + fp16 zero point per group


def fake_quantize(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Return a copy of `weight` rounded to `bits` per group of `group_size` along the last dim."""
    if bits >= NO_QUANT_BITS:
        return weight.clone()
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8 or >= {NO_QUANT_BITS}, got {bits}")
    if weight.shape[-1] % group_size:
        raise ValueError(f"last dimension {weight.shape[-1]} is not divisible by group_size {group_size}")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, group_size)
    low = grouped.amin(dim=-1, keepdim=True)
    high = grouped.amax(dim=-1, keepdim=True)
    levels = 2**bits - 1
    scale = ((high - low) / levels).clamp_min(1e-8)
    quantized = torch.round((grouped - low) / scale).clamp(0, levels)
    return (quantized * scale + low).reshape(weight.shape).to(weight.dtype)


def snapshot_experts(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """CPU copies of every layer's expert weights, so the original can be restored between runs."""
    return [(b.experts.gate_up_proj.detach().cpu().clone(), b.experts.down_proj.detach().cpu().clone())
            for b in find_moe_blocks(model)]


@torch.no_grad()
def restore_experts(model: nn.Module, snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
    blocks = find_moe_blocks(model)
    if len(blocks) != len(snapshot):
        raise ValueError(f"snapshot has {len(snapshot)} layers, model has {len(blocks)}")
    for block, (gate_up, down) in zip(blocks, snapshot):
        block.experts.gate_up_proj.copy_(gate_up)
        block.experts.down_proj.copy_(down)


@torch.no_grad()
def quantize_experts_(model: nn.Module, bits: np.ndarray, group_size: int = 128) -> None:
    """In place: quantize expert `e` of layer `l` to `bits[l, e]` (simulated). bits: [layers, experts]."""
    blocks = find_moe_blocks(model)
    bits = np.asarray(bits)
    expected = (len(blocks), blocks[0].experts.gate_up_proj.shape[0])
    if bits.shape != expected:
        raise ValueError(f"bits must have shape {expected}, got {bits.shape}")
    for block, layer_bits in zip(blocks, bits):
        for param in (block.experts.gate_up_proj, block.experts.down_proj):
            for expert, b in enumerate(layer_bits):
                param[expert].copy_(fake_quantize(param[expert], int(b), group_size))


def allocate_bits_by_coverage(score: np.ndarray, coverage_cuts: Sequence[float] = (0.6, 0.9),
                              bit_tiers: Sequence[int] = (4, 3, 2)) -> np.ndarray:
    """Per layer, rank experts by `score` (a usage share). Experts covering the first `coverage_cuts[0]`
    of usage get `bit_tiers[0]`, up to `coverage_cuts[1]` get `bit_tiers[1]`, the rest `bit_tiers[-1]`."""
    if len(bit_tiers) != len(coverage_cuts) + 1:
        raise ValueError("need exactly one more bit tier than coverage cuts")
    if list(coverage_cuts) != sorted(coverage_cuts) or not all(0 < c < 1 for c in coverage_cuts):
        raise ValueError("coverage_cuts must be increasing values in (0, 1)")
    bits = np.empty(score.shape, dtype=np.int64)
    for layer, layer_score in enumerate(score):
        order = np.argsort(-layer_score, kind="stable")
        share = layer_score[order] / layer_score.sum()
        covered_before = np.cumsum(share) - share  # usage covered by strictly more-used experts
        tier = np.searchsorted(np.asarray(coverage_cuts), covered_before, side="right")
        bits[layer, order] = np.asarray(bit_tiers)[tier]
    return bits


def allocate_like(template_bits: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Same number of experts per bit-width in every layer as `template_bits`, but ranked by `score`.
    Gives controls (other languages, random) exactly the same memory as the template."""
    bits = np.empty_like(template_bits)
    for layer, (layer_template, layer_score) in enumerate(zip(template_bits, score)):
        bits[layer, np.argsort(-layer_score, kind="stable")] = np.sort(layer_template)[::-1]
    return bits


def expert_memory_bytes(model: nn.Module, bits: np.ndarray, group_size: int = 128) -> float:
    """Bytes a real low-bit format would need for all expert weights, including scales and zero points.
    A bit-width of 0 means the expert is deleted and takes no memory."""
    blocks = find_moe_blocks(model)
    per_expert = blocks[0].experts.gate_up_proj[0].numel() + blocks[0].experts.down_proj[0].numel()
    bits = np.asarray(bits, dtype=np.float64)
    effective = np.where(bits >= NO_QUANT_BITS, NO_QUANT_BITS, bits + SCALE_ZERO_BITS / group_size)
    effective = np.where(bits == 0, 0.0, effective)
    return float(effective.sum() * per_expert / 8)
