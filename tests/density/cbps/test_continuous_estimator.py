import numpy as np
import polars as pl
import pytest
from scipy.stats import norm

from skcausal.density.cbps import CBPSContinuous


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t = 2.0 + X @ [1.0, -0.5, 0.3] + rng.normal(scale=1.5, size=n)
    return pl.DataFrame(X, schema=["x0", "x1", "x2"]), pl.DataFrame({"t": t})


def test_tags_follow_density_kind():
    assert CBPSContinuous().get_tag("density_kind") == "conditional"
    assert CBPSContinuous(density_kind="stabilized").get_tag("density_kind") == "stabilized"
    assert CBPSContinuous().get_tag("capability:t_type") == ["continuous"]
    with pytest.raises(ValueError, match="density_kind"):
        CBPSContinuous(density_kind="other")


def test_conditional_density_is_gaussian_on_original_scale():
    X, t = _data()
    estimator = CBPSContinuous().fit(X, t)
    X_new = pl.DataFrame(np.random.default_rng(1).normal(size=(8, 3)), schema=X.columns)
    t_new = pl.DataFrame({"t": np.linspace(-1, 5, 8)})

    density = estimator.predict_density(X_new, t_new)

    mean = np.column_stack([np.ones(8), X_new.to_numpy()]) @ estimator.coefficients_
    expected = norm.pdf(t_new["t"].to_numpy(), loc=mean, scale=np.sqrt(estimator.sigmasq_))
    np.testing.assert_allclose(density[:, 0], expected)
    assert estimator.coefficients_.shape == (4,)
    assert estimator.fitted_weights_.shape == (len(X),)


def test_stabilized_density_divides_by_marginal_gaussian():
    X, t = _data()
    conditional = CBPSContinuous().fit(X, t)
    stabilized = CBPSContinuous(density_kind="stabilized").fit(X, t)

    t_values = t["t"].to_numpy()
    marginal = norm.pdf(t_values, loc=t_values.mean(), scale=t_values.std(ddof=1))
    expected = conditional.predict_density(X, t)[:, 0] / marginal
    np.testing.assert_allclose(stabilized.predict_density(X, t)[:, 0], expected)


def test_fit_attributes():
    X, t = _data()
    estimator = CBPSContinuous(method="exact").fit(X, t)
    assert estimator.J_ >= 0
    assert estimator.mle_J_ >= 0
    assert isinstance(estimator.converged_, bool)
    np.testing.assert_allclose(estimator.fitted_weights_.sum(), 1.0)


def test_deterministic_treatment_falls_back_to_exact_with_warning():
    X, _ = _data()
    t = pl.DataFrame({"t": X.to_numpy() @ [1.0, -0.5, 0.3]})
    with pytest.warns(UserWarning, match="exact"):
        estimator = CBPSContinuous(method="over").fit(X, t)
    density = estimator.predict_density(X, t)
    assert np.isfinite(density).all()
