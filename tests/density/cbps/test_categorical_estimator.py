import numpy as np
import polars as pl
import pytest

from skcausal.density.cbps import CBPSCategorical


def _binary_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    treat = X @ [1.0, -0.5, 0.3] + rng.normal(size=n) > 0
    X = pl.DataFrame(X, schema=["x0", "x1", "x2"])
    return X, pl.DataFrame({"t": treat})


def _multi_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = np.column_stack([np.zeros(n), X @ [1.0, 0.0, 0.5], X @ [-0.5, 1.0, 0.0]])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (rng.uniform(size=n)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
    X = pl.DataFrame(X, schema=["x0", "x1", "x2"])
    return X, labels


def test_tags():
    estimator = CBPSCategorical()
    assert estimator.get_tag("capability:t_type") == ["categorical"]
    assert estimator.get_tag("capability:multidimensional_treatment") is False
    assert estimator.get_tag("density_kind") == "conditional"


def test_binary_density_is_probability_of_observed_level():
    X, t = _binary_data()
    estimator = CBPSCategorical().fit(X, t)

    probs = estimator.predict_proba(X)
    density = estimator.predict_density(X, t)

    assert list(estimator.classes_) == [False, True]
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    observed = np.where(t["t"].to_numpy(), probs[:, 1], probs[:, 0])
    np.testing.assert_allclose(density[:, 0], observed)
    np.testing.assert_allclose(estimator.fitted_values_, probs)
    assert estimator.coefficients_.shape == (4, 1)
    assert estimator.fitted_weights_.shape == (len(X),)


def test_att_flips_treated_level():
    X, t = _binary_data()
    att1 = CBPSCategorical(att=1).fit(X, t)
    att2 = CBPSCategorical(att=2).fit(X, t)

    treat = t["t"].to_numpy()
    np.testing.assert_allclose(att1.fitted_weights_[treat].sum(), 1.0)
    np.testing.assert_allclose(att1.fitted_weights_[~treat].sum(), 1.0)
    # ATT=1: treated units (level True) get uniform weights; ATT=2 the reverse.
    assert np.ptp(att1.fitted_weights_[treat]) < 1e-12
    assert np.ptp(att2.fitted_weights_[~treat]) < 1e-12
    assert np.ptp(att1.fitted_weights_[~treat]) > 1e-6


@pytest.mark.parametrize("dtype", ["enum", "categorical"])
def test_multi_level_treatment(dtype):
    X, labels = _multi_data()
    if dtype == "enum":
        names = np.array(["control", "low", "high"])[labels]
        t = pl.DataFrame(
            {"t": pl.Series(names).cast(pl.Enum(["control", "low", "high"]))}
        )
        expected_classes = ["control", "low", "high"]
    else:
        # polars Categorical keeps first-appearance order; sorted lexically here.
        names = np.array(["a", "b", "c"])[labels]
        t = pl.DataFrame({"t": pl.Series(names).cast(pl.Categorical)})
        present_order = list(dict.fromkeys(names))
        expected_classes = present_order

    estimator = CBPSCategorical(method="over").fit(X, t)

    assert list(estimator.classes_) == expected_classes
    probs = estimator.predict_proba(X)
    assert probs.shape == (len(X), 3)
    density = estimator.predict_density(X, t)
    np.testing.assert_allclose(density[:, 0], probs[np.arange(len(X)), labels])
    assert estimator.coefficients_.shape == (4, 2)


def test_att_rejected_for_more_than_two_levels():
    X, labels = _multi_data()
    names = np.array(["a", "b", "c"])[labels]
    t = pl.DataFrame({"t": pl.Series(names).cast(pl.Enum(["a", "b", "c"]))})
    with pytest.raises(ValueError, match="att"):
        CBPSCategorical(att=1).fit(X, t)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="method"):
        CBPSCategorical(method="nope")
    with pytest.raises(ValueError, match="att"):
        CBPSCategorical(att=3)


def test_non_numeric_covariates_raise():
    X, t = _binary_data()
    X = X.with_columns(pl.Series("cat", ["a", "b"] * (len(X) // 2)))
    with pytest.raises(ValueError, match="numeric"):
        CBPSCategorical().fit(X, t)


def test_predict_on_new_data_uses_original_scale_coefficients():
    X, t = _binary_data()
    estimator = CBPSCategorical().fit(X, t)
    X_new = pl.DataFrame(
        np.random.default_rng(3).normal(size=(10, 3)), schema=X.columns
    )

    probs = estimator.predict_proba(X_new)
    logits = (
        np.column_stack([np.ones(10), X_new.to_numpy()]) @ estimator.coefficients_[:, 0]
    )
    np.testing.assert_allclose(probs[:, 1], 1 / (1 + np.exp(-logits)), atol=1e-6)
