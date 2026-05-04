import numpy as np
import polars as pl
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from skcausal.density.permutation_weighting import PermutationWeighting


def _make_continuous_dataset(n_samples=300, random_state=42):
    X, t = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        random_state=random_state,
    )
    X = pl.DataFrame(X, schema=[f"x{i}" for i in range(X.shape[1])])
    t = pl.DataFrame({"t": t.astype(float)})
    return X, t


def _make_mixed_treatment_dataset(n_samples=160, random_state=42):
    rng = np.random.default_rng(random_state)

    X = pl.DataFrame(
        rng.normal(size=(n_samples, 3)),
        schema=["x0", "x1", "x2"],
    )
    dose = (0.8 * X["x0"].to_numpy() - 0.2 * X["x1"].to_numpy()).astype(float)
    dose += rng.normal(scale=0.4, size=n_samples)
    arm = np.where(dose > np.median(dose), "high", "low")

    t = pl.DataFrame({"dose": dose, "arm": arm}).with_columns(
        pl.col("arm").cast(pl.Categorical)
    )
    return X, t


def test_permutation_weighting_returns_stabilized_density_ratio():
    X, t = _make_continuous_dataset()

    estimator = PermutationWeighting(
        classifier=LogisticRegression(max_iter=1000),
        max_trials=1,
        random_state=0,
    )

    estimator.fit(X, t)

    density_ratio = estimator.predict_density(X, t)
    Xt = estimator._make_feature_data(X, t)
    probabilities = estimator.classifiers_[0].predict_proba(Xt)
    expected = (probabilities[:, 1] / probabilities[:, 0]).reshape(-1, 1)

    assert estimator.get_tag("density_kind") == "stabilized"
    assert density_ratio.shape == (len(X), 1)
    assert np.isfinite(density_ratio).all()
    assert (density_ratio >= 0).all()
    np.testing.assert_allclose(density_ratio, expected)


def test_permutation_weighting_fits_directly_without_weight_wrapper():
    X, t = _make_continuous_dataset(random_state=7)

    estimator = PermutationWeighting(
        classifier=LogisticRegression(max_iter=1000),
        max_trials=2,
        random_state=3,
    )

    estimator.fit(X, t)

    assert not hasattr(estimator, "weight_estimator_")
    assert estimator.classifiers_


def test_permutation_weighting_supports_convergence_fit_mode():
    X, t = _make_continuous_dataset(random_state=11)

    estimator = PermutationWeighting(
        classifier=RandomForestClassifier(n_estimators=25, random_state=0),
        max_trials=3,
        fit_mode="convergence",
        random_state=2,
    )

    estimator.fit(X, t)
    density_ratio = estimator.predict_density(X, t)

    assert estimator.classifier_ is not None
    assert density_ratio.shape == (len(X), 1)
    assert np.isfinite(density_ratio).all()


def test_permutation_weighting_supports_mixed_treatment_tables():
    X, t = _make_mixed_treatment_dataset(random_state=5)

    estimator = PermutationWeighting(
        classifier=RandomForestClassifier(n_estimators=25, random_state=0),
        max_trials=2,
        random_state=0,
    )

    estimator.fit(X, t)
    density_ratio = estimator.predict_density(X, t)

    assert estimator.get_tag("capability:multidimensional_treatment") is True
    assert density_ratio.shape == (len(X), 1)
    assert np.isfinite(density_ratio).all()
    assert (density_ratio >= 0).all()
