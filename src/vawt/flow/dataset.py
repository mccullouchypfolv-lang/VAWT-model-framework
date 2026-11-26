"""Build HF residual targets using the trained low-fidelity solver."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ResidualNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, training_residuals):
        mean = training_residuals.mean(dim=(0, 2)).unsqueeze(-1)
        std = training_residuals.std(dim=(0, 2), correction=0).unsqueeze(-1)
        if not (mean.isfinite().all() and std.isfinite().all() and (std > 0).all()):
            raise ValueError("Training residuals need a finite, positive scale in each channel")
        return cls(mean, std)

    def normalize(self, residual):
        return (residual - self.mean.to(residual)) / self.std.to(residual)

    def denormalize(self, residual):
        return residual * self.std.to(residual) + self.mean.to(residual)


class ResidualDataset(Dataset):
    """Yield physical conditions, LF and HF curves, and the standardized residual."""

    def __init__(self, conditions, lf, hf, normalizer, theta, case_ids):
        self.conditions = conditions
        self.lf = lf
        self.hf = hf
        self.normalizer = normalizer
        self.residual = normalizer.normalize(hf - lf)
        self.theta = theta
        self.case_ids = tuple(case_ids)

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, index):
        return {
            "conditions": self.conditions[index],
            "lf": self.lf[index],
            "hf": self.hf[index],
            "residual": self.residual[index],
        }


def _load_splits(path, count):
    splits = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    names = ("train", "val", "test")
    if set(splits) != set(names):
        raise ValueError("Expected train, val, and test splits")
    for name in names:
        if not isinstance(splits[name], list) or not splits[name]:
            raise ValueError(f"HF split {name!r} must be a nonempty list")
    indices = [index for name in names for index in splits[name]]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(count)):
        raise ValueError("HF splits must cover each case exactly once")
    return {name: splits[name] for name in names}


def build_residual_datasets(data_path, split_path, lf_solver, *, batch_size=64):
    """Load HF cases, generate fresh LF curves, and share train-fitted scales."""
    if batch_size < 1:
        raise ValueError("LF prediction batch size must be positive")
    with np.load(data_path, allow_pickle=False) as data:
        case_ids = data["case_ids"]
        operating = data["conditions"]
        geometry = data["geometry"]
        targets = data["target"]
        theta = torch.as_tensor(data["theta"], dtype=torch.float32)
    count = len(case_ids)
    if operating.shape != (count, 2) or geometry.shape != (count, 20, 2):
        raise ValueError("Expected wind/TSR scalars and 20-layer chord/thickness geometry")
    if targets.shape != (count, lf_solver.n_theta, 3):
        raise ValueError("Expected HF Cl, Cd, CM curves on the LF azimuth grid")
    if len(set(case_ids)) != count or not np.allclose(geometry, geometry[:, :1], rtol=1e-6, atol=1e-6):
        raise ValueError("Expected unique cases with spanwise-uniform chord and thickness")
    solver_theta = lf_solver.theta.detach().cpu()
    if theta.shape != solver_theta.shape or not torch.allclose(theta, solver_theta, rtol=0, atol=1e-6):
        raise ValueError("HF and LF must use the same azimuth grid")
    splits = _load_splits(split_path, count)
    wind, tsr = operating.T
    chord, thickness = geometry[:, 0].T
    conditions = torch.as_tensor(np.column_stack((tsr, wind, thickness, chord)), dtype=torch.float32)
    hf = torch.as_tensor(targets.transpose(0, 2, 1), dtype=torch.float32).contiguous()
    if not (conditions.isfinite().all() and hf.isfinite().all()):
        raise ValueError("HF conditions and curves must be finite")

    lf_solver.requires_grad_(False).eval()
    with torch.no_grad():
        lf = torch.cat([lf_solver(batch.to(lf_solver.theta)).cpu()
                        for batch in conditions.split(batch_size)])
    if lf.shape != hf.shape or not lf.isfinite().all():
        raise ValueError("LF solver must produce finite curves aligned with HF")
    train = splits["train"]
    normalizer = ResidualNormalizer.fit(hf[train] - lf[train])
    return {
        name: ResidualDataset(conditions[indices], lf[indices], hf[indices],
                              normalizer, theta, case_ids[indices])
        for name, indices in splits.items()
    }
