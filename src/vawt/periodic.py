"""Fourier features for angles in radians."""

import torch


def fourier_features(angles, orders):
    phase = angles.unsqueeze(-1) * orders
    return torch.cat((phase.cos(), phase.sin()), dim=-1)
