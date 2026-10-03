# NN — Assignment 2

A small NumPy neural network library (`nnlib`) and the notebooks that test it.

```
nnlib/                  the library: PLayer base class and all layers
XORTest.ipynb           part 1: solve XOR with the library
TaxiTripDuration.ipynb  part 2: predict NYC taxi trip duration
data/                   dataset (not in git, see below)
models/                 saved weights: XOR_solved.w
docs/                   assignment description, Preprocessing.md, figures/
```

## The library (`nnlib`)

| Class | Purpose |
|---|---|
| `PLayer` | Base class: every layer implements `forward()` and `backward()` |
| `LinearLayer` | Fully connected layer, `X @ W + b` |
| `SigmoidFunction`, `Tanh`, `RectifiedLinearUnit` | Activation functions |
| `BinaryCrossEntropyLoss` | Loss for binary classification (XOR) |
| `MeanSquaredErrorLoss` | Loss for regression (trip duration) |
| `Sequential` | Chains layers; `forward`, `backward`, `step(lr)`, `save(path)`, `load(path)` |

## Setup

1. Install NumPy, pandas, Matplotlib and Jupyter.
2. Download `nyc_taxi_data.npy` from Canvas and place it in `data/`.
3. Open the notebooks from the repo root so `import nnlib` works.

## Reproducing the results

Both notebooks are saved with their outputs, and every random seed is fixed, so **Run All** reproduces the same numbers.

- **`XORTest.ipynb`** (about 1 minute): trains XOR with sigmoid and with tanh hidden activations, compares them
  over 50 seeds, and saves and reloads `models/XOR_solved.w`.
- **`TaxiTripDuration.ipynb`** (about 5 minutes): preprocessing, 3 model configurations with early stopping,
  and test-set evaluation. It regenerates the figures in `docs/figures/`.
- **`docs/Preprocessing.md`**: the separate write-up of the features used and how they were transformed.
