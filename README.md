# NN — Assignment 2

A small NumPy neural network library (`nnlib`) and the notebooks that test it.

```
nnlib/                  the library: PLayer base class and all layers
XORTest.ipynb           part 1: solve XOR with the library
TaxiTripDuration.ipynb  part 2: predict NYC taxi trip duration
data/                   dataset (not in git, see below)
models/                 saved weights, e.g. XOR_solved.w
docs/                   assignment description
```

## Setup

1. Install NumPy (and Jupyter + Matplotlib for the notebooks).
2. Download `nyc_taxi_data.npy` from Canvas and place it in `data/`.
3. Open the notebooks from the repo root so `import nnlib` works.
