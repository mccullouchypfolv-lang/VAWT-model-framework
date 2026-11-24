"""Train the unmodified upstream FLORAL Flow on the fixed VAWT split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = HERE / "upstream"
sys.path.insert(0, str(UPSTREAM / "src"))
from floral.flow import Flow


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_vawt():
    selection = read(ROOT / "configs/selected_lf_model.json")
    cfg = read(ROOT / selection["config"])
    work = ROOT / cfg["work"]
    assets = work / cfg["flow_assets"]
    provenance = read(assets / "provenance.json")
    lf_path = ROOT / selection["checkpoint"]
    paired = assets / "paired_data.npz"
    split_path = assets / "splits.json"
    if digest(lf_path) != selection["checkpoint_sha256"]:
        raise ValueError("Selected LF checkpoint hash mismatch")
    for path in (lf_path, paired, split_path):
        if provenance.get(str(path.resolve())) != digest(path):
            raise ValueError(f"Paired asset provenance mismatch: {path}")
    splits = read(split_path)
    with np.load(paired, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("case_ids", "theta", "target", "lf_prediction", "conditions", "geometry")}
    if not np.allclose(arrays["geometry"], arrays["geometry"][:, :1]):
        raise ValueError("The benchmark adapter requires spanwise-uniform geometry")
    return arrays, splits, work / cfg["flow_output"], selection


def normalize(field, training):
    mean = field[training].mean((0, 2), keepdim=True)
    std = field[training].std((0, 2), keepdim=True)
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid training normalization scales")
    return (field - mean) / std, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--modes", type=int, help="Override the official FNO mode setting")
    args = parser.parse_args()
    if args.modes is not None and args.modes < 1:
        parser.error("--modes must be positive")
    arrays, splits, reference, selection = load_vawt()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    example = UPSTREAM / "examples/oneDCorr"
    config = yaml.safe_load((example / "config.yml").read_text())
    hp = yaml.safe_load((example / "hp_config.yml").read_text())
    config["floral"] = True
    if args.modes is not None:
        config["flow"]["operator"]["field"]["modes"] = args.modes
    n = len(arrays["theta"])
    lf = torch.as_tensor(arrays["lf_prediction"], dtype=torch.float32).transpose(1, 2)
    hf = torch.as_tensor(arrays["target"], dtype=torch.float32).transpose(1, 2)
    scalars = np.column_stack((arrays["conditions"], arrays["geometry"][:, 0]))
    conditions = torch.as_tensor(scalars, dtype=torch.float32)[:, :, None].expand(-1, -1, n)
    train = splits["train"]
    residual, mean, std = normalize(hf - lf, train)
    condition, _, _ = normalize(conditions, train)
    lf_normal, _, _ = normalize(lf, train)
    fields = TensorDataset(residual, condition, lf_normal)
    train_data = DataLoader(torch.utils.data.Subset(fields, train), batch_size=hp["batch_size"], shuffle=True)
    val_data = DataLoader(torch.utils.data.Subset(fields, splits["val"]), batch_size=hp["batch_size"])
    shape = {name: dict(channels=channels, dims=[n], ndim=1)
             for name, channels in (("field", 3), ("condition", 4), ("field_domain", 1), ("condition_domain", 1))}
    grid = torch.as_tensor(arrays["theta"] / (2 * np.pi), dtype=torch.float32).reshape(1, 1, n)
    model = Flow(config, hp, {"field": grid, "condition": grid}, shape)
    checkpoint = ModelCheckpoint(dirpath=out, filename="best", monitor="val_loss", mode="min")
    trainer = Trainer(
        default_root_dir=out, accelerator="gpu", devices=1, max_epochs=1 if args.smoke else hp["max_epochs"],
        gradient_clip_val=1., enable_progress_bar=False, enable_model_summary=False,
        logger=CSVLogger(str(out), name="logs"),
        callbacks=[checkpoint, EarlyStopping("val_loss", min_delta=1e-4, patience=int(.3 * hp["max_epochs"]))],
        limit_train_batches=2 if args.smoke else 1.0, limit_val_batches=1 if args.smoke else 1.0,
    )
    write(out / "run_config.json", dict(upstream="9d1c1c0", config=config, hyperparameters=hp,
          lf_checkpoint=selection["checkpoint"], lf_sha256=selection["checkpoint_sha256"],
          condition_channels=["wind_ref", "tsr", "chord", "thickness_ratio"],
          splits={name: len(ids) for name, ids in splits.items()}, smoke=args.smoke))
    trainer.fit(model, train_data, val_data)
    model = Flow.load_from_checkpoint(checkpoint.best_model_path, weights_only=False).cuda().eval()
    report = {"best_val_fm_mse": float(checkpoint.best_model_score), "epochs_run": trainer.current_epoch,
              "best_epoch": torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)["epoch"] + 1}
    for split in ("val", "test"):
        ids = splits[split][:1] if args.smoke else splits[split]
        torch.manual_seed(42 + (10000 if split == "val" else 20000))
        samples = []
        for index in ids:
            generated = model.integrate_flow(
                condition[index:index + 1].cuda(), lf_normal[index:index + 1].cuda(),
                n_gen=2 if args.smoke else 32, nT=2, method="dopri5", atol=1e-5, rtol=1e-5)
            samples.append((generated.cpu() * std + mean + lf[index]).permute(0, 1, 3, 2).numpy())
        samples = np.concatenate(samples)
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite predictions")
        prediction = samples.mean(1)
        target = arrays["target"][ids]
        np.savez_compressed(out / f"{split}_predictions.npz", case_ids=arrays["case_ids"][ids],
                            theta=arrays["theta"], target=target, mean=prediction, samples=samples,
                            lf=arrays["lf_prediction"][ids])
        if not args.smoke:
            with np.load(reference / f"{split}_predictions.npz") as previous:
                for key, expected in (("case_ids", arrays["case_ids"][ids]), ("target", target),
                                      ("theta", arrays["theta"]), ("lf", arrays["lf_prediction"][ids])):
                    np.testing.assert_array_equal(previous[key], expected)
                groups = {}
                for label, mask in (("all", np.ones(len(ids), dtype=bool)),
                                    ("tsr_le_2", arrays["conditions"][ids, 1] <= 2)):
                    truth = target[mask].astype(float)
                    rmse = np.sqrt(np.mean((prediction[mask] - truth)**2, axis=(0, 1)))
                    prev = np.sqrt(np.mean((previous["mean"][mask] - truth)**2, axis=(0, 1)))
                    relative_l2 = (np.linalg.norm(prediction[mask] - truth, axis=1)
                                   / np.linalg.norm(truth, axis=1)).mean(axis=0)
                    groups[label] = dict(n=int(mask.sum()), rmse=rmse.tolist(), reference_rmse=prev.tolist(),
                                         relative_l2_mean_per_case=relative_l2.tolist(),
                                         nrmse=(rmse / np.sqrt(np.mean(truth**2, axis=(0, 1)))).tolist())
                report[split] = groups
    write(out / "metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
