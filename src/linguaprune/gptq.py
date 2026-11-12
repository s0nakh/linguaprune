"""GPTQ for fused MoE experts (transformers 5.x layout), calibrated on chosen-language text.

GPTQ (Frantar et al., 2022) quantizes a weight matrix column by column and pushes each column's
rounding error onto the not-yet-quantized columns, using H = XᵀX of the layer's real inputs X.
For a MoE layer every expert gets its own H, built only from the calibration tokens the router
sends to that expert. Experts that receive too few tokens fall back to round-to-nearest.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import torch
from torch import nn

from linguaprune.quant import NO_QUANT_BITS, fake_quantize

ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _group_params(weight: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric min-max scale and zero point per row for one group of columns."""
    low, high = weight.amin(dim=1), weight.amax(dim=1)
    maxq = 2**bits - 1
    scale = ((high - low) / maxq).clamp_min(1e-8)
    zero = torch.round(-low / scale)
    return scale, zero


def _quantize_column(column: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    q = torch.clamp(torch.round(column / scale) + zero, 0, 2**bits - 1)
    return scale * (q - zero)


@torch.no_grad()
def gptq_quantize(weight: torch.Tensor, hessian: torch.Tensor, bits: int, group_size: int = 128,
                  damp: float = 0.01, block_size: int = 128) -> torch.Tensor:
    """Return GPTQ-quantized (dequantized) copy of `weight` [out, in] given `hessian` [in, in]."""
    if bits >= NO_QUANT_BITS:
        return weight.clone()
    W = weight.float().clone()
    H = hessian.float().clone()
    cols = W.shape[1]
    if cols % group_size or H.shape != (cols, cols):
        raise ValueError(f"weight {tuple(W.shape)} / hessian {tuple(H.shape)} / group_size {group_size} mismatch")
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W1, Q1, E1 = W[:, i1:i2].clone(), torch.zeros_like(W[:, i1:i2]), torch.zeros_like(W[:, i1:i2])
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:
                group = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :group_size]
                scale, zero = _group_params(group, bits)
            w, d = W1[:, i], Hinv1[i, i]
            q = _quantize_column(w, scale, zero, bits)
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.to(weight.dtype)


def _capture_moe_inputs(layer: nn.Module, inputs: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
                        run_layer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> list[torch.Tensor]:
    """Inputs of the MoE block for every real (non-padding) calibration token, per batch: [tokens, hidden]."""
    captured, current_mask = [], {}

    def hook(_module, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        captured.append(x[current_mask["valid"]].detach())

    handle = layer.mlp.register_forward_pre_hook(hook)
    try:
        for hidden, mask in inputs():
            current_mask["valid"] = mask.reshape(-1).bool().to(hidden.device)
            run_layer(hidden, mask)
    finally:
        handle.remove()
    return captured


@torch.no_grad()
def _expert_hessians(xs: list[torch.Tensor], gate: nn.Module, experts: nn.Module,
                     stage: str) -> tuple[list[torch.Tensor], np.ndarray]:
    """Per-expert H = XᵀX (stage 'gate_up': MoE input; stage 'down': act(gate)·up of that input)."""
    num_experts = experts.gate_up_proj.shape[0]
    dim = experts.gate_up_proj.shape[2] if stage == "gate_up" else experts.down_proj.shape[2]
    device = experts.gate_up_proj.device
    hessians = [torch.zeros(dim, dim, device=device) for _ in range(num_experts)]
    counts = np.zeros(num_experts, dtype=np.int64)
    for x in xs:
        x = x.to(device)
        _, _, chosen = gate(x)
        for e in torch.unique(chosen).tolist():
            tokens = x[(chosen == e).any(dim=1)].float()
            if stage == "down":
                g, u = nn.functional.linear(tokens, experts.gate_up_proj[e].float()).chunk(2, dim=-1)
                tokens = experts.act_fn(g) * u
            hessians[e] += tokens.T @ tokens
            counts[e] += tokens.shape[0]
    return hessians, counts


@torch.no_grad()
def quantize_layer_gptq_(layer: nn.Module, expert_bits: np.ndarray, inputs, run_layer,
                         group_size: int = 128, min_tokens: int = 16, attention_bits: int | None = 8) -> dict:
    """In place: attention to `attention_bits` (RTN), then every expert of this layer with GPTQ at
    `expert_bits[e]` bits, calibrated on `inputs()`. Returns per-expert calibration token counts."""
    if attention_bits is not None:
        for name in ATTENTION_PROJECTIONS:
            proj = getattr(layer.self_attn, name)
            proj.weight.copy_(fake_quantize(proj.weight, attention_bits, group_size))
    gate, experts = layer.mlp.gate, layer.mlp.experts
    xs = _capture_moe_inputs(layer, inputs, run_layer)
    stats = {}
    for stage, param in (("gate_up", experts.gate_up_proj), ("down", experts.down_proj)):
        hessians, counts = _expert_hessians(xs, gate, experts, stage)
        for e, bits in enumerate(expert_bits):
            bits = int(bits)
            if bits >= NO_QUANT_BITS:
                continue
            if counts[e] < min_tokens:
                param[e].copy_(fake_quantize(param[e], bits, group_size))
            else:
                param[e].copy_(gptq_quantize(param[e], hessians[e], bits, group_size))
        stats[stage] = counts
    return {"tokens_per_expert": stats["gate_up"].tolist(),
            "rtn_fallback_experts": int((stats["gate_up"] < min_tokens).sum())}
