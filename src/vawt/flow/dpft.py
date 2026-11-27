"""Dual-conditioned periodic transformer for the residual velocity field."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..periodic import fourier_features


class GlobalTimeEncoder(nn.Module):
    def __init__(self, width, global_tokens, *, time_frequency_max=1000.0):
        super().__init__()
        self.global_projection = nn.Sequential(
            nn.Linear(global_tokens * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        if not math.isfinite(time_frequency_max) or time_frequency_max < 1:
            raise ValueError("Maximum time angular frequency must be finite and at least one")
        frequencies = torch.logspace(0, math.log10(time_frequency_max), width // 2)
        self.register_buffer("time_frequencies", frequencies, persistent=False)
        self.time_projection = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, global_features, time):
        global_embedding = self.global_projection(global_features.flatten(1))
        time_embedding = self.time_projection(fourier_features(time.squeeze(-1), self.time_frequencies))
        return self.fusion(torch.cat((global_embedding, time_embedding), dim=-1))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)

    def forward(self, state, condition):
        scale, shift = self.modulation(condition).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(state) * (1 + scale) + shift


class PeriodicAttention(nn.Module):
    def __init__(self, width, heads, harmonics):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.projection = nn.Linear(width, width)
        self.relation = nn.Linear(2 * harmonics, heads, bias=False)

    def _split_heads(self, tokens):
        return tokens.unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)

    def forward(self, state, memory, relative_features):
        query = self._split_heads(self.query(state))
        key = self._split_heads(self.key(memory))
        value = self._split_heads(self.value(memory))
        bias = self.relation(relative_features).permute(2, 0, 1)
        # Aero memory is variable-major, with the same azimuth grid for each variable.
        copies = memory.shape[1] // state.shape[1]
        bias = bias.repeat(1, 1, copies).to(query)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        return self.projection(attended.transpose(1, 2).flatten(2))


class DualConditionedFlowBlock(nn.Module):
    def __init__(self, width, heads, harmonics):
        super().__init__()
        self.self_norm = AdaptiveLayerNorm(width)
        self.cross_norm = AdaptiveLayerNorm(width)
        self.ffn_norm = AdaptiveLayerNorm(width)
        self.self_attention = PeriodicAttention(width, heads, harmonics)
        self.cross_attention = PeriodicAttention(width, heads, harmonics)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, state, condition, memory, relative_features):
        normalized = self.self_norm(state, condition)
        state = state + self.self_attention(normalized, normalized, relative_features)
        state = state + self.cross_attention(
            self.cross_norm(state, condition), memory, relative_features
        )
        return state + self.ffn(self.ffn_norm(state, condition))


class DPFT(nn.Module):
    """Map CPJE features, a three-channel residual state, and FM time to velocity."""

    def __init__(self, *, time_frequency_max=1000.0):
        super().__init__()
        self.global_tokens = 4
        self.channels = 3
        self.n_theta = 160
        self.width = 48
        harmonics = 4
        self.register_buffer("orders", torch.arange(1, harmonics + 1), persistent=False)
        theta = (torch.arange(self.n_theta, dtype=torch.float64) + 0.5) * (math.tau / self.n_theta)
        self.register_buffer("theta", theta.float(), persistent=False)

        self.condition_encoder = GlobalTimeEncoder(
            self.width, self.global_tokens, time_frequency_max=time_frequency_max
        )
        self.memory_norm = nn.LayerNorm(self.width)
        self.state_conv = nn.Conv1d(self.channels, self.width, 5, padding=2, padding_mode="circular")
        self.position = nn.Linear(2 * harmonics, self.width, bias=False)
        self.blocks = nn.ModuleList(
            DualConditionedFlowBlock(self.width, heads=4, harmonics=harmonics) for _ in range(3)
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.width), nn.Linear(self.width, self.width), nn.GELU(),
            nn.Linear(self.width, self.channels),
        )

    def forward(self, global_features, aero_features, residual_state, time, *, theta=None):
        """Return velocities [B, 3, 160] in Cl, Cd, CM order."""
        if residual_state.ndim != 3 or residual_state.shape[1:] != (self.channels, self.n_theta):
            raise ValueError("Expected residual state [B, 3, 160]")
        batch = residual_state.shape[0]
        if global_features.shape != (batch, self.global_tokens, self.width):
            raise ValueError("Expected CPJE global features [B, 4, 48]")
        if aero_features.shape != (batch, self.channels, self.n_theta, self.width):
            raise ValueError("Expected CPJE aerodynamic features [B, 3, 160, 48]")
        if time.shape != (batch, 1):
            raise ValueError("Expected flow time [B, 1]")
        theta = self.theta if theta is None else theta
        if theta.shape != (self.n_theta,):
            raise ValueError("Expected shared azimuth coordinates [160] in radians")
        theta = theta.to(residual_state)
        relative_features = fourier_features(theta[:, None] - theta[None, :], self.orders)

        condition = self.condition_encoder(global_features, time)
        memory = self.memory_norm(aero_features.flatten(1, 2))
        state = self.state_conv(residual_state).transpose(1, 2)
        state = state + self.position(fourier_features(theta, self.orders))
        for block in self.blocks:
            state = block(state, condition, memory, relative_features)
        return self.decoder(state).transpose(1, 2).contiguous()
