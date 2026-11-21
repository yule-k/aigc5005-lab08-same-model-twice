"""Train the same network in Keras, with the same rows and settings as train.py."""

import time

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from keras_model import build_keras_net
from prepare import make_splits
from train import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN,
    LEARNING_RATE,
    REPORTS,
    SEED,
    plot_curves,
)


def train(seed: int = SEED, path: str | None = "bike_net.keras") -> dict:
    """Train with callbacks, keep the best epoch, and score the test rows once."""
    keras.utils.set_random_seed(seed)
    splits = make_splits(SEED)
    x, y = splits.train.x.numpy(), splits.train.y.numpy()
    x_val, y_val = splits.val.x.numpy(), splits.val.y.numpy()

    model = build_keras_net(splits.n_features, HIDDEN)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EPOCHS, restore_best_weights=True
        ),
    ]
    if path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                path, monitor="val_loss", save_best_only=True
            )
        )
    started = time.perf_counter()
    history = model.fit(
        x,
        y,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    seconds = time.perf_counter() - started

    val_curve = history.history["val_loss"]
    best_epoch = int(np.argmin(val_curve)) + 1
    predicted = model.predict(splits.test.x.numpy(), verbose=0).ravel()
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best_epoch,
        "val_mae": history.history["val_mae"][best_epoch - 1],
        "test_mae": mean_absolute_error(splits.test.y.numpy(), predicted),
        "seconds": seconds,
        "history": history.history,
    }


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    outcome = train()
    curves = outcome.pop("history")
    frame = pd.DataFrame(
        {
            "epoch": range(1, EPOCHS + 1),
            "train_loss": curves["loss"],
            "val_loss": curves["val_loss"],
            "val_mae": curves["val_mae"],
        }
    )
    frame.to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(
        frame,
        outcome["best_epoch"],
        "Keras main run",
        REPORTS / "learning_curves_keras.png",
    )
    print(outcome)
    print("Saved bike_net.keras, history_keras.csv, learning_curves_keras.png")
