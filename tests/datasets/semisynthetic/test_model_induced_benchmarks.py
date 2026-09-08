"""Tests for the named ModelInducedConfounding benchmarks.

The generic dataset contract is swept in tests/datasets/test_all_datasets.py.
What is pinned here is what each named class promises and the object sweep
cannot see: the treatment shape it claims, that rho is a live dependence knob
on the two-coordinate ones, and that the fixed regularization still leaves the
overlap the docstrings quote.

The GasDrift pair fetches its source from OpenML, so its tests sit behind the
same SKCAUSAL_TEST_OPENML gate as the rest of the network tests. Everything
else here is offline.
"""

import os

import numpy as np
import polars as pl
import pytest

from skcausal.datasets import (
    ModelInducedBreastCancerBinary,
    ModelInducedDigitsContinuous,
    ModelInducedDigitsContinuous2D,
    ModelInducedDigitsMixed,
    ModelInducedGasDriftContinuous2D,
    ModelInducedGasDriftMixed,
)

# class -> (rows, treatment columns and their polars dtype)
SHAPES = {
    ModelInducedBreastCancerBinary: (569, {"t_0": pl.Categorical}),
    ModelInducedDigitsContinuous: (1797, {"t_0": pl.Float64}),
    ModelInducedDigitsMixed: (1797, {"t_0": pl.Float64, "t_1": pl.Categorical}),
    ModelInducedDigitsContinuous2D: (1797, {"t_0": pl.Float64, "t_1": pl.Float64}),
}
COUPLED = [ModelInducedDigitsMixed, ModelInducedDigitsContinuous2D]


@pytest.mark.parametrize("cls", list(SHAPES))
def test_the_declared_treatment_shape_is_what_is_released(cls):
    rows, columns = SHAPES[cls]
    dataset = cls(random_state=0)
    covariates, treatments, outcomes = dataset.load()

    assert covariates.height == treatments.height == outcomes.height == rows
    assert treatments.columns == list(columns)
    assert dict(treatments.schema) == columns
    # The class attribute must agree with the instance, since a caller reads it
    # off the class to decide whether an estimator fits the problem.
    assert cls.column_types == dataset.column_types
    assert np.isfinite(outcomes.to_numpy()).all()


@pytest.mark.parametrize("cls", list(SHAPES))
def test_the_response_surface_is_known_over_the_grid(cls):
    dataset = cls(random_state=0)
    covariates, treatments, _ = dataset.load()
    grid = dataset.get_grid(5)

    assert dataset.predict_y(covariates, treatments).shape == (covariates.height, 1)
    assert dataset.predict_curve(covariates, grid).shape == (grid.height,)


@pytest.mark.parametrize("cls", COUPLED)
def test_rho_is_a_live_dependence_knob(cls):
    """Beyond the covariate channel the coordinates already share.

    Mutual information reads the mixed pair; for two continuous coordinates the
    signed correlation is the honest readout, since the copula moves the sign.
    """

    weak = cls(random_state=0, treatment_correlation=0.0)
    strong = cls(random_state=0, treatment_correlation=0.9)

    def measure(dataset):
        if cls is ModelInducedDigitsContinuous2D:
            return float(np.corrcoef(*dataset.load()[1].to_numpy().T)[0, 1])
        return dataset.achieved_mutual_information_[0, 1]

    assert measure(strong) > measure(weak)


def test_the_binary_benchmark_keeps_its_overlap():
    """The fixed C is the only thing standing between this and a positivity
    violation, so the overlap the docstring quotes is a test, not a comment."""

    dataset = ModelInducedBreastCancerBinary(random_state=0)
    covariates, treatments, _ = dataset.load()

    propensity = dataset.classifier_.predict_proba(covariates.to_numpy())[:, 1]

    assert np.mean((propensity > 0.05) & (propensity < 0.95)) > 0.85
    assert propensity.min() > 1e-4
    # Both levels are actually realized, or there is no treatment to contrast.
    assert set(treatments.get_column("t_0").cast(pl.Utf8).to_list()) == {"0", "1"}


def test_a_benchmark_reproduces_itself_from_its_seed():
    first = ModelInducedDigitsMixed(random_state=3).load()
    second = ModelInducedDigitsMixed(random_state=3).load()

    assert all(a.equals(b) for a, b in zip(first, second))


needs_openml = pytest.mark.skipif(
    not os.environ.get("SKCAUSAL_TEST_OPENML"),
    reason="set SKCAUSAL_TEST_OPENML=1 to fetch the gas-drift table from OpenML",
)

GAS_DRIFT_SHAPES = {
    ModelInducedGasDriftMixed: {"t_0": pl.Float64, "t_1": pl.Categorical},
    ModelInducedGasDriftContinuous2D: {"t_0": pl.Float64, "t_1": pl.Float64},
}


@needs_openml
@pytest.mark.parametrize("cls", list(GAS_DRIFT_SHAPES))
def test_the_fetched_benchmarks_are_the_large_siblings(cls):
    """13910 rows is the whole reason these two accept a download."""

    dataset = cls(random_state=0)
    covariates, treatments, outcomes = dataset.load()

    assert covariates.height == treatments.height == outcomes.height == 13910
    assert covariates.width == 128
    assert dict(treatments.schema) == GAS_DRIFT_SHAPES[cls]
    assert cls.column_types == dataset.column_types
    assert len(dataset.treatment_levels_) == 6
    assert dataset.predict_curve(covariates, dataset.get_grid(5)).shape[0] > 0


@needs_openml
@pytest.mark.parametrize("cls", list(GAS_DRIFT_SHAPES))
def test_rho_is_a_live_knob_on_the_fetched_benchmarks_too(cls):
    weak = cls(random_state=0, treatment_correlation=0.0)
    strong = cls(random_state=0, treatment_correlation=0.9)

    def measure(dataset):
        if cls is ModelInducedGasDriftContinuous2D:
            return float(np.corrcoef(*dataset.load()[1].to_numpy().T)[0, 1])
        return dataset.achieved_mutual_information_[0, 1]

    assert measure(strong) > measure(weak)
