"""Train CPJE and DPFT on standardized high-fidelity residuals."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..solver import LF_Solver
from .dataset import build_residual_datasets
from .flow import ResidualFlow


TIME_FREQUENCY_MAX = 8 * math.pi
TPC_WEIGHT = 0.25


def run_epoch(model, batches, theta, *, generator, optimizer=None, tpc_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    tpc_total = 0.0
    with torch.set_grad_enabled(training):
        for batch in batches:
            batch = {key: value.to(theta.device) for key, value in batch.items() if key != "hf"}
            fm, tpc = model.loss_terms(batch, theta, generator=generator,
                                       mirror_tpc=training and tpc_weight > 0)
            loss = fm if tpc is None else fm + tpc_weight * tpc
            value = loss.detach().item()
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite flow matching MSE")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = len(batch["conditions"])
            total += fm.detach().item() * count
            if tpc is not None:
                tpc_total += tpc.detach().item() * count
    size = len(batches.dataset)
    return total / size, tpc_total / size


def train(data_path, split_path, lf_checkpoint, output, epochs, device, *,
          batch_size=8, learning_rate=0.001, weight_decay=0.0001, seed=42, target_aware=True,
          tpc_weight=TPC_WEIGHT):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if not math.isfinite(tpc_weight) or tpc_weight < 0:
        raise ValueError("TPC weight must be finite and nonnegative")
    torch.manual_seed(seed)
    solver = LF_Solver.from_checkpoint(lf_checkpoint).to(device)
    datasets = build_residual_datasets(data_path, split_path, solver)
    training_data = datasets["train"]
    theta = training_data.theta.to(device)
    train_batches = DataLoader(
        training_data, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_batches = DataLoader(datasets["val"], batch_size=batch_size)
    train_rng = torch.Generator(device=device).manual_seed(seed)
    val_rng = torch.Generator(device=device)
    model = ResidualFlow(
        standardize_cpje_conditions=True, time_frequency_max=TIME_FREQUENCY_MAX,
        physical_bypass=True, independent_velocity_heads=True, target_aware=target_aware,
    )
    model.cpje.fit_condition_normalization(training_data.conditions)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    best_val = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        train_mse, train_tpc_mse = run_epoch(
            model, train_batches, theta, generator=train_rng, optimizer=optimizer, tpc_weight=tpc_weight,
        )
        # Reuse validation times and sources so the score tracks model changes.
        val_mse, _ = run_epoch(model, val_batches, theta, generator=val_rng.manual_seed(seed + 1))
        row = {"epoch": epoch, "train_velocity_mse": train_mse, "val_velocity_mse": val_mse}
        if tpc_weight:
            row["train_tpc_mse"] = train_tpc_mse
            row["train_total_loss"] = train_mse + tpc_weight * train_tpc_mse
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            torch.save({
                "model": model.state_dict(),
                "standardize_cpje_conditions": True,
                "dpft_blocks": len(model.dpft.blocks),
                "time_frequency_max": TIME_FREQUENCY_MAX,
                "physical_bypass": True,
                "independent_velocity_heads": True,
                "target_aware": target_aware,
                "residual_mean": training_data.normalizer.mean,
                "residual_std": training_data.normalizer.std,
                "lf_checkpoint": str(lf_checkpoint),
                "epoch": epoch,
                "tpc_weight": tpc_weight,
                "tpc_pairing": "mirror",
            }, output / "best.pt")

    report = {
        "best_epoch": best_epoch, "best_val_velocity_mse": best_val, "history": history,
        "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
        "standardize_cpje_conditions": True, "dpft_blocks": len(model.dpft.blocks),
        "time_frequency_max": TIME_FREQUENCY_MAX,
        "physical_bypass": True, "independent_velocity_heads": True,
        "target_aware": target_aware,
        "tpc_weight": tpc_weight, "tpc_pairing": "mirror",
        "weight_decay": weight_decay, "splits": {name: len(data) for name, data in datasets.items()},
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--lf-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-aware", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the default two-block target-aware velocity field")
    parser.add_argument("--tpc-weight", type=float, default=TPC_WEIGHT,
                        help="Mirror-time consistency weight (0 disables it)")
    args = parser.parse_args()
    train(args.data, args.splits, args.lf_checkpoint, args.output, args.epochs, args.device,
          batch_size=args.batch_size, learning_rate=args.learning_rate,
          weight_decay=args.weight_decay, seed=args.seed, target_aware=args.target_aware,
          tpc_weight=args.tpc_weight)


if __name__ == "__main__":
    main()
