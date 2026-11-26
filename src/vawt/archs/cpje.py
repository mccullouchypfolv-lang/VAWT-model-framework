"""Condition-aware periodic joint encoder for VAWT load curves."""

import torch
from torch import nn
from torch.nn import functional as F

from ..periodic import fourier_features


class DynamicRelationBias(nn.Module):
    """Evaluate condition-dependent Fourier biases for ordered variable pairs."""

    def __init__(self, parameters, variables, width, harmonics):
        super().__init__()
        self.parameters_count = parameters
        self.variables = variables
        self.coefficient_count = 1 + 2 * harmonics
        self.register_buffer("orders", torch.arange(1, harmonics + 1), persistent=False)
        self.coefficients = nn.Sequential(
            nn.Linear(parameters, width),
            nn.GELU(),
            nn.Linear(width, variables * variables * self.coefficient_count),
        )

    def forward(self, conditions, theta):
        # Attention rows are queries and columns are keys.
        relative_angles = theta[:, None] - theta[None, :]
        periodic = fourier_features(relative_angles, self.orders)
        basis = torch.cat((torch.ones_like(periodic[..., :1]), periodic), dim=-1)
        coefficients = self.coefficients(conditions).reshape(
            -1, self.variables, self.variables, self.coefficient_count
        )
        aerodynamic_bias = torch.einsum("bijh,pqh->bipjq", coefficients, basis.to(coefficients))
        aerodynamic_bias = aerodynamic_bias.flatten(1, 2).flatten(2, 3)
        return F.pad(aerodynamic_bias, (self.parameters_count, 0, self.parameters_count, 0))


class JointAttentionBlock(nn.Module):
    def __init__(self, width, heads, expansion):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * expansion),
            nn.GELU(),
            nn.Linear(width * expansion, width),
        )

    def forward(self, tokens, bias):
        batch, count, width = tokens.shape
        qkv = self.qkv(self.attention_norm(tokens))
        query, key, value = qkv.reshape(
            batch, count, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4).unbind()
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias[:, None].to(query)
        )
        tokens = tokens + self.projection(attended.transpose(1, 2).reshape(batch, count, width))
        return tokens + self.ffn(self.ffn_norm(tokens))


class CPJE(nn.Module):
    """Encode TSR, wind speed, thickness ratio, chord and three LF curves.

    Pass conditions [B, 4], LF curves [B, 3, N] ordered Cl, Cd, CM, and a
    shared uniform periodic grid theta [N] in radians without a repeated endpoint.
    Returns global features [B, 4, 48] and aerodynamic features [B, 3, N, 48].
    """

    def __init__(self):
        super().__init__()
        self.parameters_count = 4
        self.variables = 3
        self.width = 48
        self.kernel_size = 5
        harmonics = 4
        self.scalar_tokenizers = nn.ModuleList(
            nn.Linear(1, self.width) for _ in range(self.parameters_count)
        )
        self.parameter_embedding = nn.Parameter(torch.empty(self.parameters_count, self.width))
        self.variable_embedding = nn.Parameter(torch.empty(self.variables, self.width))
        nn.init.normal_(self.parameter_embedding, std=0.02)
        nn.init.normal_(self.variable_embedding, std=0.02)
        self.circular_encoders = nn.ModuleList(
            nn.Conv1d(
                1, self.width, self.kernel_size, padding=self.kernel_size // 2, padding_mode="circular"
            )
            for _ in range(self.variables)
        )
        self.register_buffer("orders", torch.arange(1, harmonics + 1), persistent=False)
        self.periodic_embedding = nn.Linear(2 * harmonics, self.width, bias=False)
        self.relation_bias = DynamicRelationBias(
            self.parameters_count, self.variables, self.width, harmonics
        )
        self.blocks = nn.ModuleList(
            JointAttentionBlock(self.width, heads=4, expansion=2) for _ in range(2)
        )

    def forward(self, conditions, lf_curves, theta):
        if conditions.ndim != 2 or conditions.shape[1] != self.parameters_count:
            raise ValueError("Expected conditions with shape [B, 4]")
        if lf_curves.ndim != 3 or lf_curves.shape[:2] != (conditions.shape[0], self.variables):
            raise ValueError("Expected LF curves with shape [B, 3, N]")
        batch, _, angles = lf_curves.shape
        if theta.shape != (angles,) or angles < self.kernel_size:
            raise ValueError("Expected a shared azimuth grid [N] with at least five points")
        theta = theta.to(lf_curves)

        global_tokens = torch.stack([
            tokenizer(value)
            for tokenizer, value in zip(self.scalar_tokenizers, conditions.split(1, dim=1))
        ], dim=1)
        global_tokens = global_tokens + self.parameter_embedding

        aerodynamic_tokens = torch.stack([
            encoder(curve).transpose(1, 2)
            for encoder, curve in zip(self.circular_encoders, lf_curves.split(1, dim=1))
        ], dim=1)
        periodic = self.periodic_embedding(fourier_features(theta, self.orders))
        aerodynamic_tokens = aerodynamic_tokens + self.variable_embedding[:, None] + periodic

        tokens = torch.cat((global_tokens, aerodynamic_tokens.flatten(1, 2)), dim=1)
        bias = self.relation_bias(conditions, theta)
        for block in self.blocks:
            tokens = block(tokens, bias)

        global_features = tokens[:, :self.parameters_count]
        aerodynamic_features = tokens[:, self.parameters_count:].reshape(
            batch, self.variables, angles, self.width
        )
        return global_features, aerodynamic_features
