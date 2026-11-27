"""Target-aware periodic velocity field for Cl, Cd, and CM."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..periodic import fourier_features
from .dpft import IndependentVelocityHeads, PeriodicAttention


class TargetBranch(nn.Module):
    def __init__(self, width, heads, harmonics):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.physical = nn.Linear(width, 2 * width)
        self.time = nn.Linear(width, 2 * width, bias=False)
        self.mixing = PeriodicAttention(width, heads, harmonics)
        self.lf_skip = nn.Linear(width, width)
        self.adapter = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, state, condition, time, lf, relative_features):
        scale, shift = (self.physical(condition) + self.time(time)).unsqueeze(1).chunk(2, dim=-1)
        hidden = self.norm(state) * (1 + scale) + shift
        return state + self.mixing(hidden, hidden, relative_features) + self.lf_skip(lf)


class VariableInteraction(nn.Module):
    def __init__(self, width, heads, channels):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.gate = nn.Linear(width, channels)

    def forward(self, states, condition, relation_bias):
        batch, channels, angles, width = states.shape
        tokens = self.norm(states.flatten(1, 2))
        qkv = self.qkv(tokens).reshape(batch, channels * angles, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=relation_bias[:, None])
        update = self.output(attended.transpose(1, 2).reshape(batch, channels, angles, width))
        return states + self.gate(condition).sigmoid()[:, :, None, None] * update


class TargetAwareFlowBlock(nn.Module):
    def __init__(self, width, heads, harmonics, channels):
        super().__init__()
        self.targets = nn.ModuleList(TargetBranch(width, heads, harmonics) for _ in range(channels))
        self.interaction = VariableInteraction(width, heads, channels)

    def forward(self, states, condition, time, lf, relative_features, relation_bias):
        states = torch.stack([
            branch(state, condition, time, memory, relative_features)
            for branch, state, memory in zip(self.targets, states.unbind(1), lf.unbind(1))
        ], dim=1)
        states = self.interaction(states, condition, relation_bias)
        return torch.stack([
            state + branch.adapter(state) for branch, state in zip(self.targets, states.unbind(1))
        ], dim=1)


class TargetAwareVelocity(nn.Module):
    """Process target-specific states before gated cross-variable updates."""

    def __init__(self, *, time_frequency_max):
        super().__init__()
        if not math.isfinite(time_frequency_max) or time_frequency_max < 1:
            raise ValueError("Maximum time angular frequency must be finite and at least one")
        width, channels, angles, harmonics = 48, 3, 160, 4
        theta = (torch.arange(angles, dtype=torch.float64) + 0.5) * (math.tau / angles)
        self.register_buffer("theta", theta.float(), persistent=False)
        self.register_buffer("orders", torch.arange(1, harmonics + 1), persistent=False)
        self.register_buffer("time_frequencies", torch.logspace(0, math.log10(time_frequency_max), width // 2),
                             persistent=False)
        variables = torch.arange(channels).repeat_interleave(angles)
        self.register_buffer("same_variable", variables[:, None] == variables[None, :], persistent=False)
        self.global_projection = nn.Sequential(nn.Linear(4 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.physical_projection = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.time_projection = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.state_lifting = nn.ModuleList(
            nn.Conv1d(1, width, 5, padding=2, padding_mode="circular") for _ in range(channels)
        )
        self.blocks = nn.ModuleList(TargetAwareFlowBlock(width, 4, harmonics, channels) for _ in range(2))
        self.decoder = IndependentVelocityHeads(width)

    def forward(self, global_features, aero_features, residual_state, time, *, theta, physical, relation_bias):
        theta = theta.to(residual_state)
        relative = fourier_features(theta[:, None] - theta[None, :], self.orders)
        condition = self.global_projection(global_features.flatten(1)) + self.physical_projection(physical)
        time_embedding = self.time_projection(fourier_features(time.squeeze(-1), self.time_frequencies))
        states = torch.stack([
            lifting(state).transpose(1, 2)
            for lifting, state in zip(self.state_lifting, residual_state.split(1, dim=1))
        ], dim=1)
        relation_bias = relation_bias.masked_fill(self.same_variable, float("-inf")).to(states)
        for block in self.blocks:
            states = block(states, condition, time_embedding, aero_features, relative, relation_bias)
        return torch.cat([
            head(self.decoder.norm(state)).transpose(1, 2)
            for head, state in zip(self.decoder.heads.values(), states.unbind(1))
        ], dim=1)
