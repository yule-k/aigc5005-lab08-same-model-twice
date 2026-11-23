# PyTorch or Keras for the Lakeshore bike demand model

For the manager and the two team members. Every number comes from `reports/comparison.csv`, written by `python compare.py`: the same network, the same rows, MSE loss, Adam at 3e-3, batch size 64 and 150 epochs, on seeds 42, 1 and 2.

## What I observed

1. **The results are close, slightly better in PyTorch.** Test MAE was 2.44, 2.52 and 2.49 rentals in PyTorch, and 2.75, 2.60 and 2.54 in Keras. PyTorch was lower on all three seeds, by 0.15 on average. The Keras runs alone differ by 0.21 from seed to seed, so three seeds do not prove a real difference. With copied weights the two networks agree to 6.0e-08, so the model is the same.
2. **PyTorch trained faster**, about 5 seconds a run against about 9 in Keras. For a weekly job neither matters.
3. **Keras needed less training code.** The PyTorch loop (`run_epoch` and `train_model` in `train.py`) is about 40 lines that I had to get right. In Keras the same work is one `compile` call, one `fit` call and two callbacks.
4. **The saved models differ.** Keras saved `bike_net.keras`, which holds the architecture and the weights, so one line, `keras.models.load_model`, brings the model back. PyTorch keeps a state dict, which holds only the weights, so I must first build `BikeNet` from `model.py` and then load the weights into it.

Three settings are not identical: the starting weights (each framework's default), the order of the shuffled batches, and Adam's default epsilon (1e-8 in PyTorch, 1e-7 in Keras). The seeds make each framework repeatable, not equal to the other.

## Recommendation: PyTorch

Since the accuracy is nearly the same, it is determined by the scenario. The team may try pretrained Hugging Face models later in the year, and most of those are published for PyTorch. Choosing PyTorch now means one framework for both jobs, which matters for a team of two. The loop already exists, is tested, and gave the lower error on every seed.

## What the team gives up

The shorter Keras training code, the ready-made callbacks, and a model file that carries its own architecture. The teammate who knows Keras has to learn the hand-written loop.

## What would change my recommendation

If the Hugging Face plan is dropped and the job stays this one small tabular model, I would choose Keras: less code to maintain for the same result.
