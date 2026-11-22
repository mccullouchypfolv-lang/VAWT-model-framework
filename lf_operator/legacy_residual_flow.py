"""Archived single-path velocity used before the dual-branch FLORAL migration.

Load an old checkpoint with ConditionalVelocity(**checkpoint['model_config']).
The current training entry continues to use residual_flow.ConditionalVelocity.
"""
import numbers

import torch
from torch import nn
from torch.nn import functional as F

from .model import reflect_height
from .residual_flow import SpectralConv1d, integrate_velocity


class VelocityBlock(nn.Module):
    def __init__(self, width, modes, n_conditions):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.linear = nn.Conv1d(width, width, 1)
        self.film = nn.Sequential(nn.Linear(n_conditions + 1, width), nn.GELU(),
                                  nn.Linear(width, 2 * width))

    def forward(self, x, modulation):
        gamma, beta = self.film(modulation).chunk(2, dim=-1)
        features = self.spectral(x) + self.linear(x)
        return F.gelu((1 + gamma[:, :, None]) * features + beta[:, :, None])


class ConditionalVelocity(nn.Module):
    """Original 32-channel, three-block conditional periodic velocity."""
    def __init__(self, width=32, blocks=3, modes=12, field_channels=32,
                 rotor_channels=32, height_width=8, n_conditions=2,
                 n_targets=3, lf_channels=3):
        super().__init__()
        self.config = dict(width=width, blocks=blocks, modes=modes,
                           field_channels=field_channels, rotor_channels=rotor_channels,
                           height_width=height_width, n_conditions=n_conditions,
                           n_targets=n_targets, lf_channels=lf_channels)
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 1
               for v in self.config.values()):
            raise ValueError('all model dimensions must be positive integers')
        self.height_lift = nn.Conv2d(field_channels + 1, height_width, (3, 1))
        self.lift = nn.Conv1d(n_targets + lf_channels + rotor_channels + height_width + 2, width, 1)
        self.blocks = nn.ModuleList([VelocityBlock(width, modes, n_conditions) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(),
                                   nn.Conv1d(width, n_targets, 1))

    def prepare_condition(self, lf, rotor, field, a, z, weights, theta):
        if lf.ndim != 3 or lf.shape[-1] != self.config['lf_channels']:
            raise ValueError('lf must be [B,T,lf_channels]')
        batch, nt, _ = lf.shape
        if (rotor.shape != (batch, self.config['rotor_channels'], nt)
                or field.shape != (batch, self.config['field_channels'], 20, nt)
                or a.shape != (batch, self.config['n_conditions'])
                or z.shape != (batch, 20) or weights.shape != (batch, 20)
                or theta.shape != (batch, nt)):
            raise ValueError('condition/coordinate/weight shape mismatch')
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('height weights must be finite and positive')
        height_coords = z[:, None, :, None].expand(-1, 1, -1, nt)
        encoded = F.gelu(self.height_lift(reflect_height(
            torch.cat((field, height_coords), dim=1), 1)))
        normalized_weights = weights / weights.sum(dim=-1, keepdim=True)
        pooled = (encoded * normalized_weights[:, None, :, None]).sum(dim=2)
        features = torch.cat((lf.transpose(1, 2), rotor, pooled,
                              theta.sin()[:, None], theta.cos()[:, None]), dim=1)
        return {'features': features, 'modulation': a}

    def forward(self, g, t, condition):
        if g.ndim != 3 or g.shape[-1] != self.config['n_targets']:
            raise ValueError('g must be [B,T,n_targets]')
        batch, nt, _ = g.shape
        if t.shape not in ((batch,), (batch, 1, 1)):
            raise ValueError('t must be [B] or [B,1,1]')
        features, a = condition['features'], condition['modulation']
        n_features = self.config['lf_channels'] + self.config['rotor_channels'] + self.config['height_width'] + 2
        if features.shape != (batch, n_features, nt) or a.shape != (batch, self.config['n_conditions']):
            raise ValueError('prepared condition shape mismatch')
        modulation = torch.cat((a, t.reshape(batch, 1)), dim=1)
        x = self.lift(torch.cat((g.transpose(1, 2), features), dim=1))
        for block in self.blocks:
            x = block(x, modulation)
        return self.head(x).transpose(1, 2)
