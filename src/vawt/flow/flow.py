"""Linear probability paths and conditional flow matching."""

import torch
from torch import nn
from torch.nn import functional as F

from ..archs import CPJE
from .dpft import DPFT
from .gp import sample_prior


def linear_path(source, target, time):
    """Return the interpolated residual state and its constant target velocity."""
    return torch.lerp(source, target, time.unsqueeze(-1)), target - source


class ResidualFlow(nn.Module):
    def __init__(self, *, standardize_cpje_conditions=False, time_frequency_max=1000.0,
                 physical_bypass=False, independent_velocity_heads=False):
        super().__init__()
        if physical_bypass and not standardize_cpje_conditions:
            raise ValueError("The physical bypass requires training-fitted condition standardization")
        self.physical_bypass = physical_bypass
        self.cpje = CPJE(standardize_conditions=standardize_cpje_conditions)
        self.dpft = DPFT(time_frequency_max=time_frequency_max, physical_bypass=physical_bypass,
                         independent_velocity_heads=independent_velocity_heads)

    def encode_conditions(self, conditions, lf, theta):
        global_features, aero_features = self.cpje(conditions, lf, theta)
        physical = self.cpje.normalize_conditions(conditions) if self.physical_bypass else None
        return global_features, aero_features, physical

    def forward(self, conditions, lf, state, time, theta):
        global_features, aero_features, physical = self.encode_conditions(conditions, lf, theta)
        return self.dpft(global_features, aero_features, state, time, theta=theta, physical=physical)

    def loss(self, batch, theta, *, generator=None):
        target = batch["residual"]
        source = sample_prior(target, generator=generator)
        time = torch.rand(
            (len(target), 1), device=target.device, dtype=target.dtype, generator=generator
        )
        state, velocity = linear_path(source, target, time)
        prediction = self(batch["conditions"], batch["lf"], state, time, theta)
        return F.mse_loss(prediction, velocity)
