"""Linear probability paths and conditional flow matching."""

import torch
from torch import nn
from torch.nn import functional as F

from ..archs import CPJE
from .dpft import DPFT
from .gp import sample_prior
from .target_aware import TargetAwareVelocity


def linear_path(source, target, time):
    """Return the interpolated residual state and its constant target velocity."""
    return torch.lerp(source, target, time.unsqueeze(-1)), target - source


class ResidualFlow(nn.Module):
    def __init__(self, *, standardize_cpje_conditions=False, time_frequency_max=1000.0,
                 physical_bypass=False, independent_velocity_heads=False, target_aware=False):
        super().__init__()
        if physical_bypass and not standardize_cpje_conditions:
            raise ValueError("The physical bypass requires training-fitted condition standardization")
        if target_aware and not (physical_bypass and independent_velocity_heads):
            raise ValueError("Target-aware flow requires physical bypass and independent heads")
        self.physical_bypass = physical_bypass
        self.target_aware = target_aware
        self.cpje = CPJE(standardize_conditions=standardize_cpje_conditions)
        if target_aware:
            self.dpft = TargetAwareVelocity(time_frequency_max=time_frequency_max)
        else:
            self.dpft = DPFT(time_frequency_max=time_frequency_max, physical_bypass=physical_bypass,
                             independent_velocity_heads=independent_velocity_heads)

    def encode_conditions(self, conditions, lf, theta):
        global_features, aero_features = self.cpje(conditions, lf, theta)
        physical = self.cpje.normalize_conditions(conditions) if self.physical_bypass else None
        return global_features, aero_features, physical

    def condition_velocity(self, conditions, lf, theta):
        if self.target_aware:
            global_features, aero_features, relation_bias = self.cpje(conditions, lf, theta, return_relation_bias=True)
            physical = self.cpje.normalize_conditions(conditions)

            def predict(state, time):
                return self.dpft(global_features, aero_features, state, time, theta=theta,
                                 physical=physical, relation_bias=relation_bias)
            return predict

        global_features, aero_features, physical = self.encode_conditions(conditions, lf, theta)

        def predict(state, time):
            return self.dpft(global_features, aero_features, state, time, theta=theta, physical=physical)
        return predict

    def forward(self, conditions, lf, state, time, theta):
        return self.condition_velocity(conditions, lf, theta)(state, time)

    def loss(self, batch, theta, *, generator=None):
        fm, _ = self.loss_terms(batch, theta, generator=generator)
        return fm

    def loss_terms(self, batch, theta, *, generator=None, mirror_tpc=False):
        """Compute velocity supervision and optional mirror-time consistency."""
        target = batch["residual"]
        source = sample_prior(target, generator=generator)
        time = torch.rand(
            (len(target), 1), device=target.device, dtype=target.dtype, generator=generator
        )
        state, velocity = linear_path(source, target, time)
        predict = self.condition_velocity(batch["conditions"], batch["lf"], theta)
        prediction = predict(state, time)
        fm = F.mse_loss(prediction, velocity)
        if not mirror_tpc:
            return fm, None
        mirror_time = 1 - time
        mirror_state, _ = linear_path(source, target, mirror_time)
        tpc = F.mse_loss(prediction, predict(mirror_state, mirror_time))
        return fm, tpc
