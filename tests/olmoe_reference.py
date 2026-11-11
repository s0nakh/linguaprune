"""Minimal copy of the OLMoE MoE classes from transformers 5.20.2 (Apache-2.0), used as a test fixture.

Source: transformers/models/olmoe/modeling_olmoe.py (OlmoeExperts, OlmoeTopKRouter, OlmoeSparseMoeBlock).
The `@use_experts_implementation` decorator is omitted: the eager loop below is the reference path.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class OlmoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.randn(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim, self.intermediate_dim) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
        return final_hidden_states


class OlmoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.randn(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value.to(router_logits.dtype), router_indices


class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = OlmoeTopKRouter(config)
        self.experts = OlmoeExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        return self.experts(hidden_states, top_k_index, top_k_weights).reshape(batch_size, sequence_length, hidden_dim)


class TinyMoEModel(nn.Module):
    """A stack of OLMoE MoE blocks with residual connections, plus a config like a HF model."""

    def __init__(self, num_layers=3, num_experts=16, top_k=4, hidden=32, intermediate=16, norm_topk_prob=False):
        super().__init__()
        self.config = SimpleNamespace(num_experts=num_experts, num_experts_per_tok=top_k, hidden_size=hidden,
                                      intermediate_size=intermediate, norm_topk_prob=norm_topk_prob)
        self.num_experts = num_experts
        self.layers = nn.ModuleList(
            nn.ModuleDict({"mlp": OlmoeSparseMoeBlock(self.config)}) for _ in range(num_layers)
        )

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = hidden_states + layer["mlp"](hidden_states)
        return hidden_states
