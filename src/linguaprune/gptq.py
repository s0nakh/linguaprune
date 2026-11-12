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
    """Asymmetric min-max scale and zero point per row for one group of columns. weight: [..., rows, cols]."""
    low, high = weight.amin(dim=-1), weight.amax(dim=-1)
    scale = ((high - low) / (2**bits - 1)).clamp_min(1e-8)
    return scale, torch.round(-low / scale)


def _quantize_column(column: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int) -> torch.Tensor:
    q = torch.clamp(torch.round(column / scale) + zero, 0, 2**bits - 1)
    return scale * (q - zero)


def _upper_cholesky_of_inverse(h: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """Upper Cholesky factor of H⁻¹ for one matrix, via triangular solves.

    Done one matrix at a time on purpose: batched `cholesky_inverse` goes through MAGMA's batched
    potrs, which crashed with an illegal memory access on Kaggle T4 (CUDA 13) for 64×2048×2048 inputs.
    """
    lower = torch.linalg.cholesky(h)
    lower_inv = torch.linalg.solve_triangular(lower, eye, upper=False)
    return torch.linalg.cholesky(lower_inv.T @ lower_inv, upper=True)


@torch.no_grad()
def gptq_quantize_batched(weight: torch.Tensor, hessian: torch.Tensor, bits: int, group_size: int = 128,
                          damp: float = 0.01, block_size: int = 128) -> torch.Tensor:
    """GPTQ for a batch of independent matrices at once: weight [B, out, in], hessian [B, in, in].
    Mathematically identical to running `gptq_quantize` on each pair, but loops over columns only once."""
    if bits >= NO_QUANT_BITS:
        return weight.clone()
    W = weight.float().clone()
    H = hessian.float().clone()
    batch, _, cols = W.shape
    if cols % group_size or H.shape != (batch, cols, cols):
        raise ValueError(f"weight {tuple(W.shape)} / hessian {tuple(H.shape)} / group_size {group_size} mismatch")
    diag = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = diag == 0
    diag[dead] = 1.0  # writes through the view into H
    W.masked_fill_(dead.unsqueeze(1), 0.0)
    eye = torch.eye(cols, device=H.device)
    H += (damp * diag.mean(dim=-1)).view(batch, 1, 1) * eye
    previous = torch.backends.cuda.preferred_linalg_library() if H.is_cuda else None
    if H.is_cuda:
        torch.backends.cuda.preferred_linalg_library("cusolver")  # avoid MAGMA (see _upper_cholesky_of_inverse)
    try:
        Hinv = torch.stack([_upper_cholesky_of_inverse(h, eye) for h in H])
    finally:
        if previous is not None:
            torch.backends.cuda.preferred_linalg_library(previous)
    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W1 = W[:, :, i1:i2].clone()
        Q1, E1 = torch.zeros_like(W1), torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group_size == 0:
                group = torch.cat([W1[:, :, i:], W[:, :, i2:]], dim=-1)[:, :, :group_size]
                scale, zero = _group_params(group, bits)
            w, d = W1[:, :, i], Hinv1[:, i, i].unsqueeze(1)
            q = _quantize_column(w, scale, zero, bits)
            Q1[:, :, i] = q
            err = (w - q) / d
            W1[:, :, i:] -= err.unsqueeze(2) * Hinv1[:, i, i:].unsqueeze(1)
            E1[:, :, i] = err
        Q[:, :, i1:i2] = Q1
        W[:, :, i2:] -= E1 @ Hinv[:, i1:i2, i2:]
    return Q.to(weight.dtype)


def gptq_quantize(weight: torch.Tensor, hessian: torch.Tensor, bits: int, group_size: int = 128,
                  damp: float = 0.01, block_size: int = 128) -> torch.Tensor:
    """Return a GPTQ-quantized (dequantized) copy of `weight` [out, in] given `hessian` [in, in]."""
    return gptq_quantize_batched(weight.unsqueeze(0), hessian.unsqueeze(0), bits, group_size, damp, block_size)[0]


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
                         group_size: int = 128, min_tokens: int = 16, attention_bits: int | None = 8,
                         expert_chunk: int = 32) -> dict:
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
        for bits in sorted(set(int(b) for b in expert_bits)):
            if bits >= NO_QUANT_BITS:
                continue
            chosen = [e for e, b in enumerate(expert_bits) if int(b) == bits]
            for e in (e for e in chosen if counts[e] < min_tokens):  # too little data: round to nearest
                param[e].copy_(fake_quantize(param[e], bits, group_size))
            calibrated = [e for e in chosen if counts[e] >= min_tokens]
            for start in range(0, len(calibrated), expert_chunk):  # all experts of one bit-width at once
                group = calibrated[start:start + expert_chunk]
                quantized = gptq_quantize_batched(param[group], torch.stack([hessians[e] for e in group]),
                                                  bits, group_size)
                for e, q in zip(group, quantized):
                    param[e].copy_(q)
        stats[stage] = counts
    return {"tokens_per_expert": stats["gate_up"].tolist(),
            "rtn_fallback_experts": int((stats["gate_up"] < min_tokens).sum())}


@torch.no_grad()
def router_traffic(layer: nn.Module, inputs, run_layer) -> np.ndarray:
    """How many calibration tokens the router sends to each expert of this layer (top-k picks)."""
    num_experts = layer.mlp.gate.weight.shape[0]
    counts = np.zeros(num_experts, dtype=np.int64)
    for x in _capture_moe_inputs(layer, inputs, run_layer):
        _, _, chosen = layer.mlp.gate(x)
        counts += np.bincount(chosen.reshape(-1).cpu().numpy(), minlength=num_experts)
    return counts


def make_quantization_prepare(streamed, plans: dict[str, dict], group_size: int = 128,
                              attention_bits: int | None = 8, min_tokens: int = 16,
                              log: Callable[[str], None] = print) -> tuple[Callable, dict]:
    """Build a `prepare` callback for `stream_evaluate`.

    plans[config] = {"method": "gptq" | "rtn", "bits": array [layers, experts] or a callable
    bits(config, layer_index, layer, inputs, run_layer) -> array [experts]} (the callable form lets bits
    depend on router traffic measured in the same streamed run). Configurations without a plan (e.g.
    "original") are left untouched. Returns (prepare, stats); stats collects per-layer GPTQ details.
    """
    stats: dict[str, list[dict]] = {name: [] for name in plans}

    def prepare(config: str, index: int, layer: nn.Module, inputs) -> None:
        plan = plans.get(config)
        if plan is None:
            return
        run_layer = lambda h, m: streamed.run_layer(index, h, m)
        bits = (plan["bits"](config, index, layer, inputs, run_layer) if callable(plan["bits"])
                else np.asarray(plan["bits"])[index])
        if plan["method"] == "rtn":
            if attention_bits is not None:
                for name in ATTENTION_PROJECTIONS:
                    proj = getattr(layer.self_attn, name)
                    proj.weight.copy_(fake_quantize(proj.weight, attention_bits, group_size))
            for param in (layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj):
                for e, b in enumerate(bits):
                    param[e].copy_(fake_quantize(param[e], int(b), group_size))
        elif plan["method"] == "gptq":
            layer_stats = quantize_layer_gptq_(
                layer, bits, inputs, run_layer,
                group_size=group_size, min_tokens=min_tokens, attention_bits=attention_bits)
            stats[config].append({**layer_stats, "bits": np.asarray(bits).tolist()})
            log(f"  {config}: layer {index} GPTQ, {layer_stats['rtn_fallback_experts']} experts fell back to RTN")
        else:
            raise ValueError(f"unknown method {plan['method']!r} for {config}")

    return torch.no_grad()(prepare), stats
