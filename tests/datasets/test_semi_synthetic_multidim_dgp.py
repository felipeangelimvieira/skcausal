import numpy as np
import polars as pl
import pytest
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from skcausal.datasets.semi_synthetic_multidim import (
    _conditional_gaussian,
    _exchangeable_correlation,
    _gaussian_logpdf,
    _interval_log_mass,
    _numeric_column,
    _per_component,
    _validate_levels,
    _validate_loadings,
    _validate_targets,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_per_component_broadcasts_scalars_and_checks_lengths():
    assert _per_component("x", 2.0, 3) == [2.0, 2.0, 2.0]
    assert _per_component("x", None, 2, default=1.0) == [1.0, 1.0]
    assert _per_component("x", "income", 2) == ["income", "income"]
    assert _per_component("x", [None, "a"], 2) == [None, "a"]
    with pytest.raises(ValueError, match="length n_treatments=2"):
        _per_component("x", [1.0, 2.0, 3.0], 2)


def test_component_validators():
    np.testing.assert_allclose(_validate_loadings(None, 2), [1.0, 1.0])
    np.testing.assert_allclose(_validate_loadings([1.0, 0.0], 2), [1.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        _validate_loadings([1.0, np.inf], 2)

    assert _validate_levels(None, 2) == [None, None]
    assert _validate_levels([None, 3], 2) == [None, 3]
    with pytest.raises(ValueError, match="at least 2"):
        _validate_levels([1, None], 2)
    with pytest.raises(TypeError, match="integers"):
        _validate_levels([True, None], 2)

    assert _validate_targets(None, 2) == [None, None]
    assert _validate_targets([None, "income"], 2) == [None, "income"]
    with pytest.raises(ValueError, match="repeat"):
        _validate_targets(["age", "age"], 2)
    with pytest.raises(TypeError, match="column names"):
        _validate_targets([1, None], 2)


def test_exchangeable_correlation_is_symmetric_with_unit_diagonal():
    matrix = _exchangeable_correlation(3, 0.4)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix[np.triu_indices(3, 1)], 0.4)
    np.linalg.cholesky(matrix)


def test_interval_log_mass_matches_direct_formula_and_tails():
    lower = np.array([-np.inf, -1.0, 0.5, -np.inf, 30.0, -0.2])
    upper = np.array([-0.5, 0.25, np.inf, np.inf, np.inf, 0.2])
    expected = np.log(ndtr(upper) - ndtr(lower))
    expected[4] = np.log(ndtr(-30.0))
    np.testing.assert_allclose(_interval_log_mass(lower, upper), expected, rtol=1e-12)
    # Deep tail below the double range of ndtr differences.
    assert np.isfinite(_interval_log_mass(np.array([-np.inf]), np.array([-40.0])))[0]


def test_gaussian_logpdf_matches_scipy_and_handles_empty_dimension():
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    cholesky = np.linalg.cholesky(covariance)
    residual = np.random.default_rng(1).normal(size=(4, 3, 2))
    expected = (
        multivariate_normal(mean=np.zeros(2), cov=covariance)
        .logpdf(residual.reshape(-1, 2))
        .reshape(4, 3)
    )
    np.testing.assert_allclose(_gaussian_logpdf(residual, cholesky), expected)
    assert _gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))).shape == (5,)
    assert np.all(_gaussian_logpdf(np.zeros((5, 0)), np.zeros((0, 0))) == 0.0)


def test_conditional_gaussian_matches_textbook_formula():
    covariance = _exchangeable_correlation(3, 0.3) * 4.0
    continuous = np.array([0, 2])
    categorical = np.array([1])
    regression, conditional = _conditional_gaussian(covariance, continuous, categorical)

    cov_cc = covariance[np.ix_(continuous, continuous)]
    cov_dc = covariance[np.ix_(categorical, continuous)]
    expected_regression = cov_dc @ np.linalg.inv(cov_cc)
    np.testing.assert_allclose(regression, expected_regression)
    np.testing.assert_allclose(
        conditional,
        covariance[np.ix_(categorical, categorical)] - expected_regression @ cov_dc.T,
    )

    regression, conditional = _conditional_gaussian(
        covariance, np.array([0, 1, 2]), np.array([], dtype=int)
    )
    assert regression.shape == (0, 3) and conditional.shape == (0, 0)
    regression, conditional = _conditional_gaussian(
        covariance, np.array([], dtype=int), np.array([1])
    )
    assert regression.shape == (1, 0)
    np.testing.assert_allclose(conditional, [[4.0]])


def test_numeric_column_extracts_floats_and_rejects_bad_columns():
    X = ToyRealDataset(include_missing=True).load()[0]
    values = _numeric_column(X, "age", "target column 'age'")
    assert values.shape == (8,)
    assert np.isnan(values[2])
    with pytest.raises(ValueError, match="must be numeric"):
        _numeric_column(X, "region", "target column 'region'")
    with pytest.raises(ValueError, match="two distinct finite"):
        _numeric_column(
            X.with_columns(pl.lit(1.0).alias("age")), "age", "target column 'age'"
        )
