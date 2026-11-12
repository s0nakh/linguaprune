"""Run a transformers 5.x decoder model one layer at a time, so models larger than the GPU can be
evaluated and quantized on a small GPU.

All sequences go through layer 0, their hidden states are kept, then layer 1 is loaded, and so on.
Several *configurations* (e.g. original, 4-bit, language-aware) are carried side by side: before a
configuration runs a layer, a `prepare` callback may rewrite that layer's weights (quantize them).

Evaluation never materializes full-vocabulary logits for all tokens: for each sequence we only
compute (a) the summed log-probability of a token span and (b) the logits of a few candidate tokens
at the last position. That covers perplexity, multiple-choice letters and answer scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
from torch import nn

FUSED_EXPERT_PARAMS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


@dataclass(frozen=True)
class ScoredSequence:
    """One input to score. `span` = [start, end) token positions whose log-probability is summed
    (each token predicted from the previous position); `choice_ids` = token ids whose logits are
    read at the last position."""

    ids: tuple[int, ...]
    span: tuple[int, int] | None = None
    choice_ids: tuple[int, ...] = ()


@dataclass
class SequenceScore:
    span_logprob: float = 0.0
    span_tokens: int = 0
    choice_logits: list[float] = field(default_factory=list)


class CheckpointReader:
    """Lazy access to the tensors of a safetensors checkpoint folder (single file or sharded)."""

    def __init__(self, model_dir: str | Path):
        from safetensors import safe_open

        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            with safe_open(single, framework="pt") as handle:
                self.weight_map = {key: single.name for key in handle.keys()}
        self._safe_open = safe_open
        self._handles: dict[str, object] = {}

    def __contains__(self, key: str) -> bool:
        from linguaprune.export import quantized_base

        return key in self.weight_map or quantized_base(key, self.weight_map) is not None

    def get(self, key: str) -> torch.Tensor:
        """A tensor by name; `x.weight` of a GPTQ-quantized module is returned dequantized (float32)."""
        from linguaprune.export import GPTQ_SUFFIXES, from_gptq, quantized_base

        if key not in self.weight_map:
            base = quantized_base(key, self.weight_map)
            if base is None:
                raise KeyError(f"{key} not found in checkpoint {self.model_dir}")
            return from_gptq({suffix: self._raw(f"{base}.{suffix}") for suffix in GPTQ_SUFFIXES})
        return self._raw(key)

    def _raw(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = self._safe_open(self.model_dir / shard, framework="pt")
        return self._handles[shard].get_tensor(key)


def _checkpoint_tensor(reader: CheckpointReader, full_name: str, num_experts: int) -> torch.Tensor:
    """Read one model parameter, fusing per-expert checkpoint tensors into the 3D transformers-5 layout."""
    if full_name in reader:
        return reader.get(full_name)
    base = full_name.rsplit(".", 2)[0]  # "...mlp"
    if full_name.endswith(FUSED_EXPERT_PARAMS[0]):
        gate = torch.stack([reader.get(f"{base}.experts.{e}.gate_proj.weight") for e in range(num_experts)])
        up = torch.stack([reader.get(f"{base}.experts.{e}.up_proj.weight") for e in range(num_experts)])
        return torch.cat([gate, up], dim=1)
    if full_name.endswith(FUSED_EXPERT_PARAMS[1]):
        return torch.stack([reader.get(f"{base}.experts.{e}.down_proj.weight") for e in range(num_experts)])
    raise KeyError(f"{full_name} not found in checkpoint")


def read_module_state(module: nn.Module, prefix: str, reader: CheckpointReader,
                      num_experts: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """CPU tensors for every parameter of `module`, read from the checkpoint under `prefix`."""
    return {name: _checkpoint_tensor(reader, f"{prefix}{name}", num_experts).to(dtype)
            for name, _ in module.named_parameters()}


def materialize_(module: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    """Allocate `module` on `device` and copy `state` into it (in place)."""
    module.to_empty(device=device)
    with torch.no_grad():
        for name, param in module.named_parameters():
            param.copy_(state[name])


class StreamedModel:
    """A decoder-only transformers-5 model whose layers are loaded one at a time from disk."""

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32):
        from transformers import AutoConfig, AutoModelForCausalLM

        self.device = torch.device(device)
        self.dtype = dtype
        self.config = AutoConfig.from_pretrained(model_dir)
        self.reader = CheckpointReader(model_dir)
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_config(self.config, dtype=dtype)
        self.model.eval()
        self.num_experts = int(getattr(self.config, "num_experts", 0) or 0)
        inner = self.model.model
        for prefix, module in (("model.embed_tokens.", inner.embed_tokens), ("model.norm.", inner.norm),
                               ("lm_head.", self.model.lm_head)):
            materialize_(module, read_module_state(module, prefix, self.reader, self.num_experts, dtype), self.device)
        inner.rotary_emb = type(inner.rotary_emb)(config=self.config).to(self.device)  # rebuilt off the meta device
        self.layers = inner.layers

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def read_layer_state(self, index: int) -> dict[str, torch.Tensor]:
        return read_module_state(self.layers[index], f"model.layers.{index}.", self.reader, self.num_experts, self.dtype)

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(batch.to(self.device))

    def context(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Causal mask and rotary embeddings exactly as the model's own forward builds them."""
        from transformers.masking_utils import create_causal_mask

        position_ids = torch.arange(hidden.shape[1], device=self.device).unsqueeze(0)
        mask = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=attention_mask.to(self.device),
                                  past_key_values=None, position_ids=position_ids)
        return {"attention_mask": mask, "position_ids": position_ids,
                "position_embeddings": self.model.model.rotary_emb(hidden, position_ids)}

    @torch.no_grad()
    def run_layer(self, index: int, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.layers[index](hidden, **self.context(hidden, attention_mask))

    @torch.no_grad()
    def score(self, hidden: torch.Tensor, batch: Sequence[ScoredSequence]) -> list[SequenceScore]:
        """Final norm + lm_head only at the positions each sequence needs."""
        normed = self.model.model.norm(hidden)
        scores = []
        for row, seq in enumerate(batch):
            result = SequenceScore()
            if seq.span is not None:
                start, end = seq.span
                logits = self.model.lm_head(normed[row, start - 1:end - 1]).float()
                targets = torch.tensor(seq.ids[start:end], device=logits.device)
                result.span_logprob = torch.log_softmax(logits, -1).gather(1, targets[:, None]).sum().item()
                result.span_tokens = end - start
            if seq.choice_ids:
                last = self.model.lm_head(normed[row, len(seq.ids) - 1]).float()
                result.choice_logits = last[list(seq.choice_ids)].tolist()
            scores.append(result)
        return scores


def make_batches(sequences: Sequence[ScoredSequence], max_tokens: int = 8192, pad_id: int = 0
                 ) -> list[tuple[list[int], torch.Tensor, torch.Tensor]]:
    """Group sequences of similar length; returns (original indices, right-padded ids, attention mask)."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].ids))
    batches, current = [], []
    for i in order:
        longest = max([len(sequences[j].ids) for j in current] + [len(sequences[i].ids)])
        if current and longest * (len(current) + 1) > max_tokens:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    out = []
    for members in batches:
        length = max(len(sequences[i].ids) for i in members)
        ids = torch.full((len(members), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(members), length), dtype=torch.long)
        for row, i in enumerate(members):
            n = len(sequences[i].ids)
            ids[row, :n] = torch.tensor(sequences[i].ids)
            mask[row, :n] = 1
        out.append((members, ids, mask))
    return out


PrepareFn = Callable[[str, int, nn.Module, Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]]], None]


def stream_evaluate(streamed: StreamedModel, sequences: Sequence[ScoredSequence], configs: Sequence[str],
                    prepare: PrepareFn | None = None, max_tokens: int = 8192,
                    log: Callable[[str], None] = print, layer_limit: int | None = None,
                    calibration: dict[str, Sequence[ScoredSequence]] | None = None) -> dict[str, list[SequenceScore]]:
    """Score `sequences` under every configuration, streaming one layer at a time.

    `prepare(config, layer_index, layer_module, inputs)` is called for every configuration after the
    original layer weights are loaded and before the layer runs; it may modify the weights in place.
    `inputs()` yields (hidden, attention_mask) of **that configuration's calibration sequences** at this
    layer (`calibration[config]`, never the evaluation sequences), so calibration cannot see the exam.
    `layer_limit` runs only the first N layers — for timing dry runs; the scores are then meaningless.
    """
    pad_id = int(streamed.config.pad_token_id or 0)
    calibration = calibration or {}
    eval_batches = make_batches(sequences, max_tokens, pad_id)
    calib_batches = {c: make_batches(calibration[c], max_tokens, pad_id) for c in configs if calibration.get(c)}
    hidden = {c: [streamed.embed(ids).cpu() for _, ids, _ in eval_batches] for c in configs}
    calib_hidden = {c: [streamed.embed(ids).cpu() for _, ids, _ in b] for c, b in calib_batches.items()}
    num_layers = streamed.num_layers if layer_limit is None else min(layer_limit, streamed.num_layers)
    for index in range(num_layers):
        original = streamed.read_layer_state(index)
        for config in configs:
            materialize_(streamed.layers[index], original, streamed.device)
            batches_c = calib_batches.get(config, [])
            if prepare is not None:
                def inputs(config=config, batches_c=batches_c):
                    for (_, _, mask), h in zip(batches_c, calib_hidden.get(config, [])):
                        yield h.to(streamed.device), mask
                prepare(config, index, streamed.layers[index], inputs)
            hidden[config] = [streamed.run_layer(index, h.to(streamed.device), mask).cpu()
                              for (_, _, mask), h in zip(eval_batches, hidden[config])]
            if config in calib_hidden:
                calib_hidden[config] = [streamed.run_layer(index, h.to(streamed.device), mask).cpu()
                                        for (_, _, mask), h in zip(batches_c, calib_hidden[config])]
        streamed.layers[index].to("meta")
        log(f"layer {index + 1}/{num_layers} done")
    results = {}
    for config in configs:
        scores: list[SequenceScore | None] = [None] * len(sequences)
        for (members, _, _), h in zip(eval_batches, hidden[config]):
            for i, s in zip(members, streamed.score(h.to(streamed.device), [sequences[i] for i in members])):
                scores[i] = s
        results[config] = scores
    return results
