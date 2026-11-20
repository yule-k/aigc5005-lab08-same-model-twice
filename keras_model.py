"""The same small network as BikeNet, written in Keras, and the weight copy."""

import keras
import numpy as np
import torch
from keras import layers
from torch import nn

from model import BikeNet
from prepare import make_splits

TOLERANCE = 1e-5  # the largest gap we accept between the two networks' predictions


def build_keras_net(n_features: int, hidden: int = 64) -> keras.Model:
    """Two hidden layers. Returns one raw number (rentals) per row."""
    return keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1),
        ],
        name="bike_net",
    )


def copy_torch_into_keras(torch_net: BikeNet, keras_net: keras.Model) -> None:
    """Keras keeps a kernel as (in, out), PyTorch as (out, in), so transpose."""
    linears = [m for m in torch_net.layers if isinstance(m, nn.Linear)]
    for linear, dense in zip(linears, keras_net.layers, strict=True):
        dense.set_weights(
            [linear.weight.detach().numpy().T, linear.bias.detach().numpy()]
        )


def same_weights_gap() -> float:
    """Largest difference between the two frameworks' predictions, same weights."""
    splits = make_splits(42)
    torch.manual_seed(0)
    torch_net = BikeNet(splits.n_features).eval()
    keras_net = build_keras_net(splits.n_features)
    copy_torch_into_keras(torch_net, keras_net)
    with torch.no_grad():
        a = torch_net(splits.val.x).numpy()
    b = keras_net.predict(splits.val.x.numpy(), verbose=0).ravel()
    return float(np.abs(a - b).max())
