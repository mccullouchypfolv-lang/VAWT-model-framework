"""Train LF_Solver on the prepared LF cache with standardized MSE."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..data.lf_data import load_lf_data
from .lf_solver import LF_Solver


SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def run_epoch(model, batches, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for conditions, curves in batches:
            conditions, curves = conditions.to(device), curves.to(device)
            prediction = model.predict_standardized(conditions)
            loss = F.mse_loss(prediction, model.normalize_targets(curves))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.detach().item() * len(conditions)
    mse = total / len(batches.dataset)
    if not torch.isfinite(torch.tensor(mse)):
        raise FloatingPointError("Nonfinite standardized LF loss")
    return mse


def train(cache, output, epochs, device):
    if epochs < 1:
        raise ValueError("Epochs must be positive")
    torch.manual_seed(SEED)
    conditions, curves, theta, splits = load_lf_data(cache, seed=SEED)
    dataset = TensorDataset(conditions, curves)
    batches = {
        name: DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=name == "train")
        for name, indices in splits.items()
    }
    model = LF_Solver()
    train_indices = splits["train"]
    model.fit_normalization(conditions[train_indices], curves[train_indices])
    model.theta.copy_(theta)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    output.mkdir(parents=True, exist_ok=False)
    best_mse = float("inf")
    history = []
    checkpoint = output / "best.pt"
    for epoch in range(1, epochs + 1):
        row = {
            "epoch": epoch,
            "train_mse": run_epoch(model, batches["train"], device, optimizer),
            "val_mse": run_epoch(model, batches["val"], device),
        }
        print(json.dumps(row), flush=True)
        history.append(row)
        if row["val_mse"] < best_mse:
            best_mse = row["val_mse"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, checkpoint)

    best_model = LF_Solver.from_checkpoint(checkpoint).to(device)
    report = {
        "best_epoch": best_epoch,
        "best_val_mse": best_mse,
        "test_mse": run_epoch(best_model, batches["test"], device),
        "splits": splits,
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args.cache, args.output, args.epochs, args.device)


if __name__ == "__main__":
    main()
