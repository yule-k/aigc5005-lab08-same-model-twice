"""Train both frameworks on the same rows and seeds, and save one comparison table."""

import time

import pandas as pd
import torch
from torch import nn

from keras_model import TOLERANCE, same_weights_gap
from model import BikeNet
from prepare import make_loaders, make_splits
from train import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN,
    LEARNING_RATE,
    REPORTS,
    SEED,
    run_epoch,
    train_model,
)
from train_keras import train as train_keras

SEEDS = (42, 1, 2)


def train_torch(seed: int = SEED) -> dict:
    """The Lab 07 loop with the main settings. Scores the test rows once."""
    torch.manual_seed(seed)
    splits = make_splits(SEED)
    train_loader, val_loader, test_loader = make_loaders(splits, BATCH_SIZE, seed)
    model = BikeNet(splits.n_features, HIDDEN)

    started = time.perf_counter()
    history, best_epoch = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )
    seconds = time.perf_counter() - started

    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())
    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": history[best_epoch - 1]["val_mae"],
        "test_mae": test_mae,
        "seconds": seconds,
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    gap = same_weights_gap()
    print(f"Same weights, largest prediction gap: {gap:.1e} (tolerance {TOLERANCE})")

    rows = []
    for seed in SEEDS:
        rows.append(train_torch(seed))
        keras_run = train_keras(seed, path=None)
        keras_run.pop("history")
        rows.append(keras_run)

    frame = pd.DataFrame(rows).round({"val_mae": 3, "test_mae": 3, "seconds": 1})
    frame.to_csv(REPORTS / "comparison.csv", index=False)
    print(frame.to_string(index=False))
    print("Saved comparison.csv")


if __name__ == "__main__":
    main()
