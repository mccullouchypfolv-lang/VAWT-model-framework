"""Generate high-fidelity curves by integrating the residual velocity field."""

import torch

from .dataset import ResidualNormalizer
from .flow import ResidualFlow
from .gp import sample_prior


def rk4(velocity, initial_state, *, steps=128):
    """Integrate a velocity function from flow time zero to one."""
    if steps < 1:
        raise ValueError("RK4 requires at least one step")
    state = initial_state
    time = state.new_tensor(0.0)
    step = 1.0 / steps
    for index in range(steps):
        time = time.new_tensor(index * step)
        k1 = velocity(time, state)
        k2 = velocity(time + step / 2, state + step / 2 * k1)
        k3 = velocity(time + step / 2, state + step / 2 * k2)
        k4 = velocity(time + step, state + step * k3)
        state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def load_flow_checkpoint(path, *, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualFlow(
        standardize_cpje_conditions=checkpoint.get("standardize_cpje_conditions", False),
        time_frequency_max=checkpoint.get("time_frequency_max", 1000.0),
        physical_bypass=checkpoint.get("physical_bypass", False),
        independent_velocity_heads=checkpoint.get("independent_velocity_heads", False),
    )
    model.load_state_dict(checkpoint["model"])
    normalizer = ResidualNormalizer(checkpoint["residual_mean"], checkpoint["residual_std"])
    return model.to(device).eval(), normalizer


@torch.no_grad()
def generate_hf(model, lf_solver, conditions, normalizer, *, steps=128, generator=None):
    """Return one HF draw per condition, shaped [B, 3, 160]."""
    lf = lf_solver(conditions)
    theta = lf_solver.theta
    global_features, aero_features, physical = model.encode_conditions(conditions, lf, theta)
    source = sample_prior(lf, generator=generator)

    def velocity(time, state):
        return model.dpft(global_features, aero_features, state,
                          time.expand(len(source), 1), theta=theta, physical=physical)

    residual = normalizer.denormalize(rk4(velocity, source, steps=steps))
    return lf + residual
