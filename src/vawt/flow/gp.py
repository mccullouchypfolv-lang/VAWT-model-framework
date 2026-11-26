"""Independent standard Gaussian source for residual flow matching."""

import torch


def sample_prior(reference, *, generator=None):
    """Draw N(0, I) with the reference tensor's shape, device, and dtype."""
    return torch.randn(
        reference.shape, device=reference.device, dtype=reference.dtype, generator=generator
    )
