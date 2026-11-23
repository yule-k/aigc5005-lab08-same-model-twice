# Lakeshore Bike Demand: the same model in PyTorch and Keras


## What it does

This project predicts how many bikes will be rented from a station in one hour, from the hour, the weather, the day type and the station type. It trains a small PyTorch network with a hand-written training loop and compares it with two simple baselines: the training mean and a linear regression. Lab 08 adds a second version of the same network in Keras. The project checks that the two versions compute the same function, trains both with the same rows and settings on three seeds, and compares them. The recommendation for the team is in [COMPARISON.md](COMPARISON.md).


## How to run it

```bash
conda env create -f environment.yml
conda activate lab08-same-model-twice
python train.py          # PyTorch main run
python train_keras.py    # Keras main run
python compare.py        # both frameworks
python -m pytest
python -m ruff format --check .
python -m ruff check .
```
The three scripts recreate the files in `reports/`. `python compare.py` trains six models.

`python train_keras.py` also saves the best model as `bike_net.keras`. This checkpoint is ignored by `.gitignore` and not committed.

The repository is named `aigc5005-lab08-same-model-twice` instead of `lab08-same-model-twice`; the files and the layout are the same.

## Data and split

`bikes_data.py` builds 6,000 station-hours. The rows are split 60/20/20 into training (3,600), validation (1,200) and test (1,200) rows with `train_test_split` and `random_state=42`, before any preparation.

The five numeric columns are standardised with `StandardScaler` and `station_type` is one-hot encoded, giving 8 features. Both are fitted on the training rows only and then applied to the validation and test rows, so no information from those rows leaks into training. All settings were chosen on the validation rows; the test rows were used once, at the end.

## Model and training

- **Network** (`model.py`): 8 inputs → 64 → ReLU → 64 → ReLU → 1 output, with no activation at the end, because the target is a count, not a probability.
- **Loss**: mean squared error (MSE). The mean absolute error (MAE) is reported because it is easier to read.
- **Optimizer**: Adam, learning rate 3e-3.
- **Batches**: 64 rows; only the training loader is shuffled.
- **Epochs**: 150. After training, the weights of the epoch with the lowest validation loss are put back (epoch 106).
- **Seeds**: `torch.manual_seed(42)` and a seeded `torch.Generator` for the training DataLoader, set in `train.py`. Running `python train.py` twice on the same computer gives identical numbers.

## Results

Mean absolute error, in rentals (lower is better):

| Model | Validation MAE | Test MAE |
|---|---|---|
| Training mean | 10.15 | 9.65 |
| Linear regression | 9.32 | 8.57 |
| **Network** | **2.63** | **2.44** |

The network's error is about a quarter of the baselines' errors. The linear model does barely better than the mean because rentals do not rise in a straight line with the hour: weekdays have two commute peaks (around 8:00 and 17:30), weekends have one midday peak, and park stations behave the opposite way to downtown stations. A straight line cannot bend to follow these peaks or change shape with the day and the station type.

![Learning curves of the main run](reports/learning_curves.png)

Both losses fall quickly in the first epochs (validation loss 199 at epoch 1) and then flatten. The validation loss is lowest at epoch 106 (15.7). After that the training loss keeps falling slowly while the validation loss creeps up, the first sign of overfitting, which is why the best weights are put back (numbers from `reports/history.csv`).

## Experiments

Each run changes one setting from the main run, with the same seed.

| Run | Change | Best epoch | Best val MAE | Final train loss | Final val loss |
|---|---|---|---|---|---|
| main | none | 106 | 2.63 | 9.20 | 17.23 |
| lr_tiny | learning rate 3e-3 → 1e-5 | 150 | 9.25 | 194.38 | 226.67 |
| lr_huge | learning rate 3e-3 → 1e-1 | 89 | 2.96 | 16.32 | 25.63 |
| wide | hidden size 64 → 1024 | 26 | 2.61 | 5.41 | 21.55 |

![Curves-lr_tiny:](reports/curves_lr_tiny.png)
- **lr_tiny – underfitting.** Both losses are still high and still falling at the last epoch, so the best epoch is the last one. The steps are too small to learn anything in 150 epochs; its MAE is no better than the linear model.  

![Curves-lr_huge:](reports/curves_lr_huge.png)
- **lr_huge – unstable.** The curves are jagged and jump up and down, because each step is too large and overshoots. It ends worse than the main run.  

![Curves-wide:](reports/curves_wide.png)
- **wide – overfitting.** With about 1.06 million weights instead of about 4,800, it reaches its best validation loss already at epoch 26. After that the training loss keeps falling while the validation loss rises (to 21.6): it memorises the training rows. Its best MAE (2.61) is almost the same as the main run's, so the smaller network is kept.  

![Learning curves of the main run](reports/learning_curves.png)
- **main – fine.** Both losses fall together and only drift apart slightly at the end.

## The Keras version

**What I built.** `keras_model.py` builds the same network with `keras.Sequential`: 8 inputs → 64 → ReLU → 64 → ReLU → 1 output, 4,801 parameters, with a configurable hidden size. `train_keras.py` trains it with `compile` and `fit`, using the same rows, MSE loss, Adam at 3e-3, batch size 64, 150 epochs and seed 42 as `train.py`. The settings are imported from `train.py`, and the arrays come from the same `make_splits` call, so the data is not prepared a second time. `EarlyStopping(restore_best_weights=True)` puts back the best epoch, with a patience equal to the number of epochs so the run never stops early, and `ModelCheckpoint` saves the best model. Every epoch is logged in `reports/history_keras.csv`.

![Learning curves of the Keras main run](reports/learning_curves_keras.png)

The Keras main run reaches its lowest validation loss at epoch 96 (validation MAE 2.86, test MAE 2.75).

**The equivalence check.** `copy_torch_into_keras` in `keras_model.py` copies the weights of a `BikeNet` into the Keras network. A Keras kernel is stored as (inputs, outputs) and a PyTorch weight as (outputs, inputs), so each matrix is transposed. With the same weights, the largest difference between the two sets of predictions on the 1,200 validation rows is 6.0e-08, inside the tolerance of 1e-5. `python compare.py` prints this number.

**Results over three seeds** (`reports/comparison.csv`):

| Framework | Seed | Best epoch | Validation MAE | Test MAE | Seconds |
|---|---|---|---|---|---|
| PyTorch | 42 | 106 | 2.625 | 2.440 | 6.2 |
| Keras | 42 | 96 | 2.857 | 2.748 | 9.3 |
| PyTorch | 1 | 69 | 2.617 | 2.515 | 4.6 |
| Keras | 1 | 54 | 2.720 | 2.601 | 9.4 |
| PyTorch | 2 | 143 | 2.641 | 2.487 | 4.6 |
| Keras | 2 | 99 | 2.711 | 2.538 | 9.4 |

PyTorch has the lower test MAE on all three seeds, by 0.15 rentals on average, and trains in about half the time. The gap is about as large as the spread between the Keras seeds, so it is not proof of a real difference. The full comparison and the recommendation are in [COMPARISON.md](COMPARISON.md).

## Tests

`python -m pytest` runs 21 tests: the 11 from Lab 07 and 10 for the Keras version and the comparison (`tests/test_keras.py`, eight test functions, one of them parametrized over three hidden sizes). The new tests check that:

- the Keras network returns one value per row
- both networks have the same parameter count for hidden sizes 8, 64 and 256
- after the weight copy the two networks predict the same values to within 1e-5
- Keras gets the same rows as PyTorch
- the Keras network can memorize one small batch
- a saved and reloaded Keras model gives the same predictions
- the best weights are restored after training
- the same seed repeats a short run.

## Limitations

- The data are invented by `bikes_data.py`, not taken from a real city. The model has learned the rules of that generator, not real riders.
- It sees only six columns. It does not know the date, season, holidays, events, the rentals in the previous hour or how many bikes are actually in the dock, so it cannot predict demand that bikes were not there to meet.
- It gives one number with no range of uncertainty.
- A planner should not use it to move bikes in a real city without retraining and testing it on real data from that city.
- The comparison uses three seeds, and the two frameworks do not start from the same weights, shuffle the batches in the same order, or use the same default epsilon in Adam. Small differences between them should not be read as one framework being more accurate.
