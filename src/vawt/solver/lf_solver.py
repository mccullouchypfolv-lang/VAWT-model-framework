"""Residual MLP surrogate for single-blade low-fidelity load curves."""

import math

import torch
from torch import nn


def _scale(values, dim):
    scale = values.std(dim=dim, correction=0, keepdim=True)
    return torch.where(scale > 0, scale, torch.ones_like(scale))


class ResidualMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 256),
        )

    def forward(self, features):
        return features + self.ffn(features)


class LF_Solver(nn.Module):
    """Map [TSR, wind speed, thickness ratio, chord] to [Cl, Cd, CM] curves."""

    target_names = ("Cl", "Cd", "CM")
    n_theta = 160

    def __init__(self):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(1, 4))
        self.register_buffer("input_scale", torch.ones(1, 4))
        self.register_buffer("target_mean", torch.zeros(1, 3, 1))
        self.register_buffer("target_scale", torch.ones(1, 3, 1))
        angles = (torch.arange(self.n_theta, dtype=torch.float64) + 0.5) * (math.tau / self.n_theta)
        self.register_buffer("theta", angles.float())
        self.input_projection = nn.Sequential(
            nn.Linear(4, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU()
        )
        self.backbone = nn.Sequential(*(ResidualMLPBlock() for _ in range(3)))
        self.decoders = nn.ModuleDict({
            target: nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, self.n_theta))
            for target in self.target_names
        })

    @torch.no_grad()
    def fit_normalization(self, conditions, curves):
        if conditions.ndim != 2 or conditions.shape[1] != 4 or not len(conditions):
            raise ValueError("Expected nonempty training conditions [B, 4]")
        if curves.shape != (len(conditions), len(self.target_names), self.n_theta):
            raise ValueError("Expected training curves [B, 3, 160]")
        if not (conditions.isfinite().all() and curves.isfinite().all()):
            raise ValueError("Training data must be finite")
        self.input_mean.copy_(conditions.mean(dim=0, keepdim=True))
        self.input_scale.copy_(_scale(conditions, dim=0))
        self.target_mean.copy_(curves.mean(dim=(0, 2), keepdim=True))
        self.target_scale.copy_(_scale(curves, dim=(0, 2)))

    def normalize_targets(self, curves):
        return (curves - self.target_mean) / self.target_scale

    def predict_standardized(self, conditions):
        if conditions.ndim != 2 or conditions.shape[1] != 4:
            raise ValueError("Expected conditions [B, 4] in TSR, wind, thickness, chord order")
        normalized = (conditions - self.input_mean) / self.input_scale
        features = self.backbone(self.input_projection(normalized))
        return torch.stack([self.decoders[target](features) for target in self.target_names], dim=1)

    def forward(self, conditions):
        """Return dimensionless coefficient curves [B, 3, 160] for CPJE."""
        return self.predict_standardized(conditions) * self.target_scale + self.target_mean

    @classmethod
    def from_checkpoint(cls, path):
        model = cls()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        return model.eval()
