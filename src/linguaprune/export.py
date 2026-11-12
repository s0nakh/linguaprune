"""Write language-aware GPTQ models as real low-bit checkpoints in the standard GPTQ ("v1") layout.

Every quantized linear layer `<name>.weight` [out, in] becomes four tensors, as written by AutoGPTQ /
GPTQModel and read by vLLM and transformers:
- `<name>.qweight` int32 [in * bits / 32, out]: codes packed along the input dimension;
- `<name>.qzeros`  int32 [groups, out * bits / 32]: zero points minus one, packed along the output;
- `<name>.scales`  fp16  [groups, out];
- `<name>.g_idx`   int32 [in]: group of each input column (no act-order, so i // group_size).
Codes are packed as a little-endian bit stream: value i of a column occupies bits [i*bits, (i+1)*bits).

Experts of the same layer may have different bit widths (4/3); each module's width is declared in
`quantization_config.dynamic` (GPTQModel convention) and is also recoverable from tensor shapes.

`convert_to_gptq_v2` rewrites a checkpoint into GPTQModel's "gptq_v2" layout, which is identical except that
`qzeros` stores the zero points themselves instead of zero minus one. GPTQModel refuses asymmetric v1
checkpoints from other producers (older tools wrote overflowing zero points), but loads v2 directly.
`CheckpointReader` (linguaprune.streaming) dequantizes these tensors transparently, so a written
checkpoint can be evaluated by the same streaming code that produced it.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import torch
from torch import nn

GPTQ_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")
ZERO_OFFSET = {"gptq": 1, "gptq_v2": 0}  # stored qzeros = zero point - offset
WORD_BITS = 32
COPIED_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "LICENSE")


def pack_rows(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unsigned integer codes [n, m] (each < 2**bits) along dim 0 into int32 [n * bits / 32, m].
    Every 32 codes fill exactly `bits` words, so the work is vectorized over those blocks."""
    n, m = codes.shape
    if n % WORD_BITS:
        raise ValueError(f"number of packed rows {n} must be a multiple of {WORD_BITS}")
    codes = codes.to(torch.int64)
    if codes.min() < 0 or codes.max() >= 2**bits:
        raise ValueError(f"codes out of range for {bits} bits")
    blocks = codes.reshape(n // WORD_BITS, WORD_BITS, m)
    words = torch.zeros(n // WORD_BITS, bits + 1, m, dtype=torch.int64)  # +1 spill word, always empty at the end
    for j in range(WORD_BITS):
        word, shift = divmod(j * bits, WORD_BITS)
        words[:, word] |= (blocks[:, j] << shift) & 0xFFFFFFFF
        if shift + bits > WORD_BITS:
            words[:, word + 1] |= blocks[:, j] >> (WORD_BITS - shift)
    words = words[:, :bits].reshape(n * bits // WORD_BITS, m)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of `pack_rows`: int32 [n * bits / 32, m] -> int64 codes [n, m]."""
    rows, m = packed.shape
    if rows % bits:
        raise ValueError(f"{rows} packed rows is not a whole number of {bits}-word blocks")
    words = (packed.to(torch.int64) & 0xFFFFFFFF).reshape(rows // bits, bits, m)
    words = torch.cat([words, torch.zeros_like(words[:, :1])], dim=1)
    mask = 2**bits - 1
    values = []
    for j in range(WORD_BITS):
        word, shift = divmod(j * bits, WORD_BITS)
        value = words[:, word] >> shift
        if shift + bits > WORD_BITS:
            value = value | (words[:, word + 1] << (WORD_BITS - shift))
        values.append(value & mask)
    return torch.stack(values, dim=1).reshape(rows // bits * WORD_BITS, m)


def to_gptq(weight: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int,
            group_size: int) -> dict[str, torch.Tensor]:
    """GPTQ tensors for one already-quantized linear weight [out, in] on the grid (scale, zero) [out, groups].
    Raises if the weight is not exactly on that grid, so a checkpoint can never silently differ from the
    model that was evaluated."""
    out_features, in_features = weight.shape
    groups = in_features // group_size
    if scale.shape != (out_features, groups) or zero.shape != (out_features, groups):
        raise ValueError(f"scale/zero {tuple(scale.shape)} do not match weight {tuple(weight.shape)}")
    if scale.dtype != torch.float32 or not torch.equal(scale, scale.half().float()):
        raise ValueError("scale must be float32 values exactly representable in fp16")
    w = weight.detach().float().cpu().reshape(out_features, groups, group_size)
    s, z = scale.cpu()[..., None], zero.cpu()[..., None]
    codes = torch.round(w / s + z)
    if codes.min() < 0 or codes.max() > 2**bits - 1 or not torch.equal(s * (codes - z), w):
        raise ValueError("weight is not exactly on its quantization grid")
    if zero.min() < 1 or zero.max() > 2**bits - 1:
        raise ValueError("zero points must lie in [1, 2**bits - 1] for the v1 format")
    return {
        "qweight": pack_rows(codes.reshape(out_features, in_features).T.contiguous(), bits),
        "qzeros": pack_rows(zero.cpu().to(torch.int64) - 1, bits).T.contiguous(),  # [out, groups] packed along out
        "scales": scale.cpu().T.contiguous().half(),
        "g_idx": torch.arange(in_features, dtype=torch.int32) // group_size,
    }


def gptq_bits(tensors: dict[str, torch.Tensor]) -> int:
    """Bit width of a GPTQ module, from its tensor shapes."""
    in_features = tensors["g_idx"].shape[0]
    bits, remainder = divmod(tensors["qweight"].shape[0] * WORD_BITS, in_features)
    if remainder:
        raise ValueError("qweight shape does not match g_idx")
    return bits


def from_gptq(tensors: dict[str, torch.Tensor], checkpoint_format: str = "gptq") -> torch.Tensor:
    """Dequantize GPTQ tensors back to a float32 weight [out, in] (exactly the evaluated values)."""
    if checkpoint_format not in ZERO_OFFSET:
        raise ValueError(f"unknown checkpoint_format {checkpoint_format!r}; expected one of {sorted(ZERO_OFFSET)}")
    bits = gptq_bits(tensors)
    g_idx = tensors["g_idx"].to(torch.int64)
    in_features = g_idx.shape[0]
    scales = tensors["scales"].float()  # [groups, out]
    groups, out_features = scales.shape
    group_size = in_features // groups
    if not torch.equal(g_idx, torch.arange(in_features) // group_size):
        raise ValueError("act-order checkpoints (permuted g_idx) are not supported")
    codes = unpack_rows(tensors["qweight"], bits).T.float().reshape(out_features, groups, group_size)
    zeros = (unpack_rows(tensors["qzeros"].T.contiguous(), bits).T + ZERO_OFFSET[checkpoint_format]).float()
    s, z = scales.T[..., None], zeros.T[..., None]
    return (s * (codes - z)).reshape(out_features, in_features)


def _dynamic_rules(layer_bits: dict[int, list[int]], attention_bits: int | None, default_bits: int) -> dict:
    """GPTQModel `dynamic` overrides: attention width and every expert whose width differs from default."""
    rules = {}
    if attention_bits is not None and attention_bits != default_bits:
        rules[r"+:.*\.self_attn\..*"] = {"bits": attention_bits}
    for index, bits in sorted(layer_bits.items()):
        for width in sorted(set(bits) - {default_bits}):
            experts = "|".join(str(e) for e, b in enumerate(bits) if b == width)
            rules[rf"+:model\.layers\.{index}\.mlp\.experts\.({experts})\..*"] = {"bits": width}
    return rules


class GPTQCheckpointWriter:
    """Writes one model, one decoder layer per shard, while the streamed run quantizes it.

    Use `write_layer` as the `on_quantized` callback of `make_quantization_prepare` (for one configuration),
    then call `finish` once every layer is written."""

    def __init__(self, out_dir: str | Path, source_reader, group_size: int = 128,
                 default_bits: int = 4, attention_bits: int | None = 8, description: dict | None = None):
        """`source_reader`: a `CheckpointReader` of the original model; unquantized tensors are copied from it."""
        self.out_dir = Path(out_dir)
        self.source = source_reader
        self.source_dir = Path(source_reader.model_dir)
        self.group_size = group_size
        self.default_bits = default_bits
        self.attention_bits = attention_bits
        self.description = description or {}
        self.weight_map: dict[str, str] = {}
        self.layer_bits: dict[int, list[int]] = {}
        self.total_bytes = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, tensors: dict[str, torch.Tensor], shard: str) -> None:
        from safetensors.torch import save_file

        save_file(tensors, self.out_dir / shard, metadata={"format": "pt"})
        self.weight_map.update({name: shard for name in tensors})
        self.total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())

    def _add_linear(self, tensors: dict, name: str, weight: torch.Tensor, entry) -> None:
        bits, scale, zero = entry
        for suffix, value in to_gptq(weight, scale, zero, bits, self.group_size).items():
            tensors[f"{name}.{suffix}"] = value

    @torch.no_grad()
    def write_layer(self, config: str, index: int, layer: nn.Module, record: dict) -> None:
        prefix = f"model.layers.{index}."
        tensors: dict[str, torch.Tensor] = {}
        for name, entry in record["attention"].items():
            self._add_linear(tensors, f"{prefix}self_attn.{name}", getattr(layer.self_attn, name).weight, entry)
        experts = layer.mlp.experts
        num_experts = experts.gate_up_proj.shape[0]
        missing = [e for e in range(num_experts) if e not in record["gate_up"] or e not in record["down"]]
        if missing:
            raise ValueError(f"layer {index}: experts {missing[:5]}... were not quantized; cannot export")
        intermediate = experts.down_proj.shape[2]
        for e in range(num_experts):
            bits, scale, zero = record["gate_up"][e]
            fused = experts.gate_up_proj[e]
            base = f"{prefix}mlp.experts.{e}."
            self._add_linear(tensors, base + "gate_proj", fused[:intermediate],
                             (bits, scale[:intermediate], zero[:intermediate]))
            self._add_linear(tensors, base + "up_proj", fused[intermediate:],
                             (bits, scale[intermediate:], zero[intermediate:]))
            self._add_linear(tensors, base + "down_proj", experts.down_proj[e], record["down"][e])
        quantized = {f"self_attn.{n}.weight" for n in record["attention"]} | {"mlp.experts.gate_up_proj",
                                                                              "mlp.experts.down_proj"}
        for name, _ in layer.named_parameters():
            if name not in quantized:  # norms, router gate: copied unchanged from the original checkpoint
                tensors[prefix + name] = self.source.get(prefix + name).contiguous()
        self.layer_bits[index] = [int(record["gate_up"][e][0]) for e in range(num_experts)]
        self._save(tensors, f"model-layer-{index:03d}.safetensors")

    def finish(self, num_layers: int) -> dict:
        """Copy the unquantized non-layer tensors, config, tokenizer files and write the index."""
        if sorted(self.layer_bits) != list(range(num_layers)):
            raise ValueError(f"only layers {sorted(self.layer_bits)} were written, expected {num_layers}")
        rest = {key: self.source.get(key).contiguous() for key in self.source.weight_map
                if not key.startswith("model.layers.")}
        self._save(rest, "model-rest.safetensors")
        config = json.loads((self.source_dir / "config.json").read_text())
        config["quantization_config"] = {
            "quant_method": "gptq", "bits": self.default_bits, "group_size": self.group_size,
            "desc_act": False, "sym": False, "checkpoint_format": "gptq", "true_sequential": True,
            "dynamic": _dynamic_rules(self.layer_bits, self.attention_bits, self.default_bits),
        }
        (self.out_dir / "config.json").write_text(json.dumps(config, indent=2))
        for name in COPIED_FILES:
            if (self.source_dir / name).exists():
                shutil.copy(self.source_dir / name, self.out_dir / name)
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": dict(sorted(self.weight_map.items()))}
        (self.out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        summary = {**self.description, "group_size": self.group_size, "attention_bits": self.attention_bits,
                   "expert_bits_per_layer": [self.layer_bits[i] for i in range(num_layers)],
                   "tensor_bytes": self.total_bytes}
        (self.out_dir / "linguaprune.json").write_text(json.dumps(summary, indent=2))
        return summary


def quantized_base(key: str, weight_map: dict) -> str | None:
    """'x.weight' -> 'x' if the checkpoint stores x as GPTQ tensors, else None."""
    if key.endswith(".weight") and re.sub(r"\.weight$", ".qweight", key) in weight_map:
        return key[: -len(".weight")]
    return None


def qzeros_v1_to_v2(qzeros: torch.Tensor, bits: int) -> torch.Tensor:
    """v1 qzeros (zero - 1, packed along the output) -> v2 qzeros (zero itself, same packing)."""
    zeros = unpack_rows(qzeros.T.contiguous(), bits) + 1  # [out, groups]
    if zeros.max() > 2**bits - 1:
        raise ValueError(f"zero point {int(zeros.max())} does not fit {bits} bits")
    return pack_rows(zeros, bits).T.contiguous()


def convert_to_gptq_v2(model_dir: str | Path) -> int:
    """Rewrite a v1 GPTQ checkpoint folder into the gptq_v2 layout, in place, one shard at a time.
    Only `qzeros` and `quantization_config.checkpoint_format` change; returns the number of modules converted."""
    import os

    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    quant = config.get("quantization_config", {})
    if quant.get("checkpoint_format", "gptq") != "gptq":
        raise ValueError(f"{model_dir} is not a v1 GPTQ checkpoint (checkpoint_format={quant.get('checkpoint_format')!r})")
    index = model_dir / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    converted = 0
    for shard in shards:
        with safe_open(model_dir / shard, framework="pt") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
            metadata = handle.metadata()
        for key in [k for k in tensors if k.endswith(".qzeros")]:
            base = key[: -len(".qzeros")]
            if f"{base}.qweight" not in tensors or f"{base}.g_idx" not in tensors:
                raise ValueError(f"{base}: qweight/g_idx must be in the same shard as qzeros")
            bits = gptq_bits({"qweight": tensors[f"{base}.qweight"], "g_idx": tensors[f"{base}.g_idx"]})
            tensors[key] = qzeros_v1_to_v2(tensors[key], bits)
            converted += 1
        temporary = model_dir / f"{shard}.v2-tmp"
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, model_dir / shard)
    config["quantization_config"] = {**quant, "checkpoint_format": "gptq_v2"}
    config_path.write_text(json.dumps(config, indent=2))
    return converted
