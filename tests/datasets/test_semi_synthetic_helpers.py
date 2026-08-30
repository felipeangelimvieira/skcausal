import numpy as np
import polars as pl
import pytest

from skcausal.datasets.semi_synthetic import (
    _fit_baseline_surface,
    _fit_causal_score_map,
)
from tests.datasets._semi_synthetic_test_utils import ToyRealDataset


def test_score_map_is_frozen_finite_and_handles_unknown_categories():
    X = ToyRealDataset(include_missing=True).load()[0]
    score_map, scores = _fit_causal_score_map(X, np.random.default_rng(4))
    repeated = score_map.transform(X)
    new_X = X.with_columns(pl.lit("west").alias("region"))
    unseen = score_map.transform(new_X)

    for name in ("confounders", "prognostic", "treatment_only", "modifiers"):
        np.testing.assert_allclose(getattr(scores, name), getattr(repeated, name))
        assert np.isfinite(getattr(unseen, name)).all()
    assert scores.confounders.shape == (X.height, 3)


def test_baseline_is_standardized_and_exposes_targeted_selection_signal():
    X = ToyRealDataset().load()[0]
    _, scores = _fit_causal_score_map(X, np.random.default_rng(8))
    surface = _fit_baseline_surface(scores, np.random.default_rng(9))
    mean, signal = surface.evaluate(scores)

    np.testing.assert_allclose(mean.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(mean.std(ddof=0), 1.0, atol=1e-12)
    assert mean.shape == signal.shape == (X.height,)


def test_score_map_rejects_changed_schema_and_all_missing_numeric_column():
    X = ToyRealDataset().load()[0]
    score_map, _ = _fit_causal_score_map(X, np.random.default_rng(2))

    with pytest.raises(ValueError, match="columns"):
        score_map.transform(X.drop("income"))
    with pytest.raises(ValueError, match="finite observed value"):
        _fit_causal_score_map(
            X.with_columns(pl.lit(None, dtype=pl.Float64).alias("income")),
            np.random.default_rng(2),
        )
