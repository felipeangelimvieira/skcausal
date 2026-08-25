"""Generate the CSV datasets used for R compatibility fixtures.

Run from the repository root::

    uv run python tests/density/cbps/fixtures/make_fixture_data.py

Then produce the R reference outputs with ``generate_fixtures.R``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
N = 300


def _covariates(rng):
    X = rng.normal(size=(N, 4))
    X[:, 3] = rng.uniform(-2, 2, size=N)
    return X


def binary():
    rng = np.random.default_rng(10)
    X = _covariates(rng)
    treat = (X @ [0.8, -0.5, 0.3, 0.4] + rng.normal(size=N) > 0).astype(int)
    return X, treat


def three_level():
    rng = np.random.default_rng(20)
    X = _covariates(rng)
    logits = np.column_stack(
        [np.zeros(N), X @ [0.8, 0.0, 0.4, -0.3], X @ [-0.4, 0.7, 0.0, 0.3]]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=N)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    return X, np.array(["a", "b", "c"])[labels]


def four_level():
    rng = np.random.default_rng(30)
    X = _covariates(rng)
    logits = np.column_stack(
        [
            np.zeros(N),
            X @ [0.7, 0.0, 0.3, -0.3],
            X @ [-0.4, 0.6, 0.0, 0.3],
            X @ [0.2, -0.3, 0.5, 0.0],
        ]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=N)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    return X, np.array(["a", "b", "c", "d"])[labels]


def continuous():
    rng = np.random.default_rng(40)
    X = _covariates(rng)
    t = 1.5 + X @ [0.6, -0.4, 0.2, 0.3] + rng.normal(scale=1.2, size=N)
    return X, t


def main():
    for name, maker in {
        "binary": binary,
        "three_level": three_level,
        "four_level": four_level,
        "continuous": continuous,
    }.items():
        X, treat = maker()
        frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        frame.insert(0, "treat", treat)
        frame.to_csv(HERE / f"{name}.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
