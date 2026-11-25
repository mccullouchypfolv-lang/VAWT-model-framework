"""Read the prepared, parameter-only low-fidelity curve cache."""

from pathlib import Path

import numpy as np
import torch

from lf_operator.blade_data import CONDITIONS, FEATURES, INPUT_CONTRACT, TARGET_CONTRACT, TARGETS
from lf_operator.data import group_split, read_json


def load_lf_data(cache, *, seed):
    cache = Path(cache)
    manifest = read_json(cache / "manifest.json")
    expected = {
        "input_contract": INPUT_CONTRACT, "target_contract": TARGET_CONTRACT,
        "features": list(FEATURES), "conditions": list(CONDITIONS), "targets": list(TARGETS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected a parameter-only single-blade Cl/Cd/CM LF cache")

    records = manifest["records"]
    splits = group_split(records, seed=seed)
    conditions, curves = [], []
    theta = None
    for record in records:
        path = cache / record["file"]
        with np.load(path, allow_pickle=False) as sample:
            wind_tsr, geometry, target, angles = (sample[key] for key in ("a", "x", "y", "theta"))
        if (wind_tsr.shape != (2,) or geometry.shape != (20, 2, 160)
                or target.shape != (160, 3) or angles.shape != (160,)):
            raise ValueError(f"{path}: incompatible LF tensor dimensions")
        if not np.allclose(geometry, geometry[:1, :, :1], rtol=1e-6, atol=1e-6):
            raise ValueError(f"{path}: expected one chord and thickness ratio for the blade")
        if theta is None:
            theta = angles
        elif not np.array_equal(theta, angles):
            raise ValueError(f"{path}: azimuth grid differs across LF cases")
        wind, tsr = wind_tsr
        chord, thickness = geometry[0, :, 0]
        conditions.append((tsr, wind, thickness, chord))
        curves.append(target.T)

    data = tuple(torch.as_tensor(np.stack(values), dtype=torch.float32) for values in (conditions, curves))
    if not all(tensor.isfinite().all() for tensor in data):
        raise ValueError("LF cache contains nonfinite values")
    return *data, torch.as_tensor(theta, dtype=torch.float32), splits
