import keras
import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_error

from keras_model import TOLERANCE, build_keras_net, copy_torch_into_keras
from model import BikeNet
from prepare import make_loaders, make_splits
from train import SEED
from train_keras import train


def test_keras_net_returns_one_value_per_row(splits):
    model = build_keras_net(splits.n_features)
    out = model.predict(splits.val.x.numpy(), verbose=0)
    assert out.shape == (len(splits.val), 1)


@pytest.mark.parametrize("hidden", [8, 64, 256])
def test_both_networks_have_the_same_parameter_count(splits, hidden):
    torch_count = sum(
        p.numel() for p in BikeNet(splits.n_features, hidden).parameters()
    )
    assert build_keras_net(splits.n_features, hidden).count_params() == torch_count


def test_same_weights_give_the_same_predictions(splits):
    torch.manual_seed(0)
    torch_net = BikeNet(splits.n_features).eval()
    keras_net = build_keras_net(splits.n_features)
    copy_torch_into_keras(torch_net, keras_net)
    with torch.no_grad():
        expected = torch_net(splits.val.x).numpy()
    actual = keras_net.predict(splits.val.x.numpy(), verbose=0).ravel()
    assert np.allclose(expected, actual, atol=TOLERANCE)


def test_keras_and_pytorch_get_the_same_rows(splits):
    keras_splits = make_splits(SEED)
    keras_parts = (keras_splits.train, keras_splits.val, keras_splits.test)
    loaders = make_loaders(splits)
    for part, loader in zip(keras_parts, loaders, strict=True):
        assert np.array_equal(part.x.numpy(), loader.dataset.x.numpy())
        assert np.array_equal(part.y.numpy(), loader.dataset.y.numpy())


def test_keras_can_memorize_one_small_batch(splits):
    keras.utils.set_random_seed(0)
    x, y = splits.train.x.numpy()[:16], splits.train.y.numpy()[:16]
    model = build_keras_net(splits.n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss=keras.losses.MeanSquaredError(),
    )
    start = model.evaluate(x, y, verbose=0)
    history = model.fit(x, y, epochs=300, batch_size=16, verbose=0)
    assert history.history["loss"][-1] < 0.05 * start


def test_keras_model_survives_save_and_load(splits, tmp_path):
    model = build_keras_net(splits.n_features)
    path = tmp_path / "net.keras"
    model.save(path)
    loaded = keras.models.load_model(path)
    x = splits.val.x.numpy()
    assert np.allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0))


def test_best_weights_are_restored(splits, tmp_path):
    path = tmp_path / "best.keras"
    outcome = train(seed=1, path=str(path), epochs=60)
    best_val_loss = min(outcome["history"]["val_loss"])

    saved = keras.models.load_model(path)
    x_val, y_val = splits.val.x.numpy(), splits.val.y.numpy()
    val_loss, _ = saved.evaluate(x_val, y_val, verbose=0)
    assert val_loss == pytest.approx(best_val_loss, rel=1e-4)

    predicted = saved.predict(splits.test.x.numpy(), verbose=0).ravel()
    test_mae = mean_absolute_error(splits.test.y.numpy(), predicted)
    assert test_mae == pytest.approx(outcome["test_mae"], rel=1e-4)


def test_same_seed_repeats_a_short_run():
    first = train(seed=1, path=None, epochs=3)
    again = train(seed=1, path=None, epochs=3)
    assert first["history"]["val_loss"] == again["history"]["val_loss"]
    assert first["test_mae"] == again["test_mae"]
