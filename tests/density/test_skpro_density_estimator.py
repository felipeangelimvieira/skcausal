import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.density.skpro import SkproDensityEstimator


skpro = pytest.importorskip("skpro")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skpro.regression.residual import ResidualDouble


def _make_synthetic_dataset(n_samples=400, random_state=42):
    rng = np.random.default_rng(random_state)

    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    noise_scale = 0.2 + 0.5 * np.abs(x1)
    noise = rng.normal(scale=noise_scale, size=n_samples)
    t = 1.5 * x0 - 0.8 * x1 + noise

    X = pd.DataFrame({"x0": x0, "x1": x1})
    y = pd.DataFrame({"t": t})
    return X, y


def _make_estimator(random_state=0):
    return ResidualDouble(
        estimator=LinearRegression(),
        estimator_resid=RandomForestRegressor(
            n_estimators=25,
            random_state=random_state,
        ),
    )


def _negative_log_density_loss(density):
    density = np.asarray(density, dtype=float)
    density = np.clip(density, 1e-12, None)
    return -np.mean(np.log(density))


def test_skpro_density_estimator_matches_direct_skpro_loss():
    X, t = _make_synthetic_dataset()
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        t,
        test_size=0.3,
        random_state=0,
    )

    direct_estimator = _make_estimator(random_state=0)
    wrapped_estimator = SkproDensityEstimator(estimator=_make_estimator(random_state=0))

    direct_estimator.fit(X_train, t_train)
    wrapped_estimator.fit(pl.from_pandas(X_train), pl.from_pandas(t_train))

    direct_density = (
        direct_estimator.predict_proba(X_test).pdf(t_test).to_numpy(dtype=float)
    )
    wrapped_density = wrapped_estimator.predict_density(
        pl.from_pandas(X_test),
        pl.from_pandas(t_test),
    )

    direct_loss = _negative_log_density_loss(direct_density)
    wrapped_loss = _negative_log_density_loss(wrapped_density)

    assert np.allclose(wrapped_density, direct_density)
    assert wrapped_loss == pytest.approx(direct_loss)
