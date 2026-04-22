import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.density.compose import CompositeFactorizedDensityEstimator
from skcausal.density.sklearn import SklearnCategoricalDensity
from skcausal.density.skpro import SkproDensityEstimator


pytest.importorskip("skpro")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from skpro.regression.residual import ResidualDouble


def _make_synthetic_dataset(n_samples=500, random_state=42):
    rng = np.random.default_rng(random_state)

    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)

    t_cont_1 = 1.5 * x0 - 0.3 * x1 + rng.normal(scale=0.5, size=n_samples)
    t_cont_2 = -0.2 * x0 + 0.9 * x2 + rng.normal(scale=0.3, size=n_samples)
    logits = 0.7 * x0 - 1.1 * x1 + 0.2 * x2
    t_disc = (logits + rng.normal(scale=0.5, size=n_samples) > 0).astype(bool)

    X = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})
    t = pd.DataFrame({"t_cont_1": t_cont_1, "t_disc": t_disc, "t_cont_2": t_cont_2})
    return X, t


def _make_continuous_estimator(random_state=0):
    return ResidualDouble(
        estimator=LinearRegression(),
        estimator_resid=RandomForestRegressor(
            n_estimators=25,
            random_state=random_state,
        ),
    )


def test_composed_density_estimator_matches_independent_product():
    X, t = _make_synthetic_dataset()
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        t,
        test_size=0.3,
        random_state=0,
    )

    composed_estimator = CompositeFactorizedDensityEstimator(
        continuous_estimator=_make_continuous_estimator(random_state=0),
        classifier=LogisticRegression(max_iter=1000),
    )
    composed_estimator.fit(pl.from_pandas(X_train), pl.from_pandas(t_train))

    continuous_1 = SkproDensityEstimator(_make_continuous_estimator(random_state=0))
    continuous_2 = SkproDensityEstimator(_make_continuous_estimator(random_state=0))
    categorical = SklearnCategoricalDensity(LogisticRegression(max_iter=1000))

    X_train_pl = pl.from_pandas(X_train)
    X_test_pl = pl.from_pandas(X_test)
    t_train_pl = pl.from_pandas(t_train)
    t_test_pl = pl.from_pandas(t_test)

    continuous_1.fit(X_train_pl, t_train_pl.select("t_cont_1"))
    continuous_2.fit(X_train_pl, t_train_pl.select("t_cont_2"))
    categorical.fit(X_train_pl, t_train_pl.select("t_disc"))

    density_1 = continuous_1.predict_density(X_test_pl, t_test_pl.select("t_cont_1"))
    density_2 = continuous_2.predict_density(X_test_pl, t_test_pl.select("t_cont_2"))
    observed_probability = categorical.predict_density(
        X_test_pl,
        t_test_pl.select("t_disc"),
    )

    direct_density = density_1 * observed_probability * density_2
    composed_density = composed_estimator.predict_density(X_test_pl, t_test_pl)

    assert np.allclose(composed_density, direct_density)
    assert composed_density.shape == (len(X_test), 1)
    assert np.isfinite(composed_density).all()
    assert (composed_density >= 0).all()
