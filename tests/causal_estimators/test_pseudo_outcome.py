import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from skcausal.causal_estimators.pseudo_outcome import DoublyRobustPseudoOutcome
from skcausal.datasets import Synthetic2MultidimDataset
from skcausal.density.base import BaseDensityEstimator
from skcausal.density.permutation_weighting import PermutationWeighting


class TrainingMeanOutcomeRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_y_ = np.asarray(y, dtype=float).reshape(-1)
        self.mean_ = float(self.fit_y_.mean())
        return self

    def predict(self, X):
        self.predict_X_ = copy.deepcopy(X)
        return np.full(len(X), self.mean_, dtype=float)


class RecordingPseudoOutcomeRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_y_ = np.asarray(y, dtype=float).reshape(-1)
        return self

    def predict(self, X):
        self.predict_X_ = copy.deepcopy(X)
        return np.zeros(len(X), dtype=float)


class TrainingMeanDensityEstimator(BaseDensityEstimator):
    _tags = {
        "backend": "pandas",
        "density_kind": "stabilized",
    }

    def _fit(self, X, t):
        self.fit_X_ = copy.deepcopy(X)
        self.fit_t_ = copy.deepcopy(t)
        self.density_ = float(X["x"].mean() + 1.0)
        return self

    def _predict_density(self, X, t):
        self.predict_X_ = copy.deepcopy(X)
        self.predict_t_ = copy.deepcopy(t)
        return np.full((len(X), 1), self.density_, dtype=float)


def _make_mixed_treatment_regressor(categorical_column):
    return Pipeline(
        [
            (
                "encode_categorical",
                ColumnTransformer(
                    [
                        (
                            "categorical_treatment",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                            [categorical_column],
                        )
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=25,
                    min_samples_leaf=4,
                    random_state=0,
                ),
            ),
        ]
    )


def _make_training_data():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    t = pd.DataFrame({"t": [10.0, 20.0, 30.0, 40.0]})
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 5.0]})
    return X, t, y


def _make_estimator(cv, cross_fit=False):
    return DoublyRobustPseudoOutcome(
        density_estimator=TrainingMeanDensityEstimator(),
        outcome_regressor=TrainingMeanOutcomeRegressor(),
        pseudo_outcome_regressor=RecordingPseudoOutcomeRegressor(),
        cv=cv,
        cross_fit=cross_fit,
        random_state=0,
    )


def _expected_pseudo_outcomes(
    X: pd.DataFrame, y: pd.DataFrame, cv, cross_fit=False
) -> np.ndarray:
    observed_outcomes, observed_density = _expected_nuisance_predictions(
        X,
        y,
        cv=cv,
        cross_fit=cross_fit,
    )
    anchor_mean_outcomes = observed_outcomes.copy()
    return (
        y["y"].to_numpy(dtype=float) - observed_outcomes
    ) / observed_density + anchor_mean_outcomes


def _expected_nuisance_predictions(
    X: pd.DataFrame, y: pd.DataFrame, cv, cross_fit=False
) -> tuple[np.ndarray, np.ndarray]:
    x_values = X["x"].to_numpy(dtype=float)
    y_values = y["y"].to_numpy(dtype=float)
    n_samples = len(X)

    if cv is None or cv <= 1:
        if cross_fit:
            full_idx = np.arange(n_samples, dtype=int)
            inner_splitter = KFold(n_splits=2, shuffle=True, random_state=0)
            outcome_positions, density_positions = next(inner_splitter.split(full_idx))
            outcome_train_idx = full_idx[outcome_positions]
            density_train_idx = full_idx[density_positions]
            observed_outcomes = np.full(
                n_samples,
                y_values[outcome_train_idx].mean(),
                dtype=float,
            )
            observed_density = np.full(
                n_samples,
                x_values[density_train_idx].mean() + 1.0,
                dtype=float,
            )
        else:
            observed_outcomes = np.full(n_samples, y_values.mean(), dtype=float)
            observed_density = np.full(n_samples, x_values.mean() + 1.0, dtype=float)
    else:
        observed_outcomes = np.empty(n_samples, dtype=float)
        observed_density = np.empty(n_samples, dtype=float)
        splitter = KFold(n_splits=cv, shuffle=True, random_state=0)
        for train_idx, test_idx in splitter.split(X):
            if cross_fit:
                inner_splitter = KFold(n_splits=2, shuffle=True, random_state=0)
                outcome_positions, density_positions = next(
                    inner_splitter.split(train_idx)
                )
                outcome_train_idx = train_idx[outcome_positions]
                density_train_idx = train_idx[density_positions]
            else:
                outcome_train_idx = train_idx
                density_train_idx = train_idx

            observed_outcomes[test_idx] = y_values[outcome_train_idx].mean()
            observed_density[test_idx] = x_values[density_train_idx].mean() + 1.0

    return observed_outcomes, observed_density


@pytest.mark.parametrize("disabled_cv", [None, 0, 1])
def test_disabled_cv_uses_full_sample_nuisance_predictions(disabled_cv):
    X, t, y = _make_training_data()

    estimator = _make_estimator(cv=disabled_cv, cross_fit=False)
    estimator.fit(X, t, y)

    np.testing.assert_allclose(
        estimator.pseudo_outcome_regressor_.fit_y_,
        _expected_pseudo_outcomes(X, y, cv=0),
    )


@pytest.mark.parametrize("disabled_cv", [None, 0, 1])
def test_cross_fit_without_cv_splits_full_dataset_once_without_refitting_nuisances(
    disabled_cv,
):
    X, t, y = _make_training_data()
    expected_design = pd.concat(
        [X.reset_index(drop=True), t.reset_index(drop=True)],
        axis=1,
    )

    estimator = _make_estimator(cv=disabled_cv, cross_fit=True)
    estimator.fit(X, t, y)

    expected_cross_fit = _expected_pseudo_outcomes(X, y, cv=disabled_cv, cross_fit=True)
    expected_full_sample = _expected_pseudo_outcomes(
        X, y, cv=disabled_cv, cross_fit=False
    )

    np.testing.assert_allclose(
        estimator.pseudo_outcome_regressor_.fit_y_,
        expected_cross_fit,
    )
    assert not np.allclose(expected_cross_fit, expected_full_sample)
    assert len(estimator.nuisance_models_) == 1
    assert not estimator.outcome_regressor_.fit_X_.equals(expected_design)
    assert estimator.outcome_regressor_.fit_X_.shape[0] < expected_design.shape[0]
    assert not estimator.density_estimator_.fit_X_.equals(X)
    assert estimator.density_estimator_.fit_X_.shape[0] < X.shape[0]


def test_cross_fitting_uses_oof_nuisance_predictions_without_refitting_full_models():
    X, t, y = _make_training_data()

    estimator = _make_estimator(cv=2)
    estimator.fit(X, t, y)

    expected_cross_fit = _expected_pseudo_outcomes(X, y, cv=2)
    expected_no_cross_fit = _expected_pseudo_outcomes(X, y, cv=0)

    np.testing.assert_allclose(
        estimator.pseudo_outcome_regressor_.fit_y_,
        expected_cross_fit,
    )
    assert not np.allclose(expected_cross_fit, expected_no_cross_fit)
    assert len(estimator.nuisance_models_) == 2
    assert not hasattr(estimator, "density_estimator_")
    assert not hasattr(estimator, "outcome_regressor_")


def test_cross_fit_uses_separate_halves_for_outcome_and_density_nuisances():
    X, t, y = _make_training_data()

    estimator = _make_estimator(cv=2, cross_fit=True)
    estimator.fit(X, t, y)

    expected_cross_fit = _expected_pseudo_outcomes(X, y, cv=2, cross_fit=True)
    expected_shared_fold_fit = _expected_pseudo_outcomes(X, y, cv=2, cross_fit=False)

    np.testing.assert_allclose(
        estimator.pseudo_outcome_regressor_.fit_y_,
        expected_cross_fit,
    )
    assert not np.allclose(expected_cross_fit, expected_shared_fold_fit)


def test_anchor_mean_uses_the_model_for_the_anchor_held_out_fold():
    X, t, y = _make_training_data()

    estimator = _make_estimator(cv=2, cross_fit=False)
    estimator.fit(X, t, y)

    observed_outcomes, observed_density = _expected_nuisance_predictions(
        X,
        y,
        cv=2,
        cross_fit=False,
    )
    expected_anchor_specific = _expected_pseudo_outcomes(X, y, cv=2, cross_fit=False)
    blended_anchor_mean = np.full(
        len(X),
        observed_outcomes.mean(),
        dtype=float,
    )
    blended_pseudo_outcomes = (
        y["y"].to_numpy(dtype=float) - observed_outcomes
    ) / observed_density + blended_anchor_mean

    np.testing.assert_allclose(
        estimator.pseudo_outcome_regressor_.fit_y_,
        expected_anchor_specific,
    )
    assert not np.allclose(expected_anchor_specific, blended_pseudo_outcomes)


@pytest.mark.parametrize(
    ("cv", "cross_fit", "expected_error", "message"),
    [
        (True, False, TypeError, "cv must be an integer or None."),
        (-1, False, ValueError, "cv must be greater than or equal to 0 when provided."),
        (2, "yes", TypeError, "cross_fit must be a boolean."),
    ],
)
def test_cv_validation(cv, cross_fit, expected_error, message):
    with pytest.raises(expected_error, match=message):
        _make_estimator(cv=cv, cross_fit=cross_fit)


def test_cross_fit_requires_two_samples_per_cv_training_fold():
    X = pd.DataFrame({"x": [0.0, 1.0]})
    t = pd.DataFrame({"t": [10.0, 20.0]})
    y = pd.DataFrame({"y": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match="cross_fit=True requires at least 2 samples in each nuisance-training split.",
    ):
        _make_estimator(cv=2, cross_fit=True).fit(X, t, y)


def test_cross_fit_without_cv_requires_two_samples():
    X = pd.DataFrame({"x": [0.0]})
    t = pd.DataFrame({"t": [10.0]})
    y = pd.DataFrame({"y": [1.0]})

    with pytest.raises(
        ValueError,
        match="cross_fit=True requires at least 2 samples in each nuisance-training split.",
    ):
        _make_estimator(cv=0, cross_fit=True).fit(X, t, y)


def test_pseudo_outcome_supports_mixed_treatment_with_permutation_weighting():
    dataset = Synthetic2MultidimDataset(
        n=96,
        n_features=4,
        n_categorical_treatments=2,
        mutual_info=0.7,
        categorical_effect_scale=0.2,
        random_state=0,
    )
    X, t, y = dataset.load()
    _, categorical_column = t.columns

    estimator = DoublyRobustPseudoOutcome(
        density_estimator=PermutationWeighting(
            classifier=RandomForestClassifier(
                n_estimators=25,
                min_samples_leaf=4,
                random_state=0,
            ),
            max_trials=5,
            random_state=0,
        ),
        outcome_regressor=_make_mixed_treatment_regressor(categorical_column),
        pseudo_outcome_regressor=_make_mixed_treatment_regressor(categorical_column),
        cv=2,
        cross_fit=True,
        n_pseudo_samples=48,
        random_state=0,
    )

    estimator.fit(X, t, y)
    response = np.asarray(estimator.predict(dataset.get_grid(4)), dtype=float).reshape(
        -1
    )

    assert response.shape == (8,)
    assert np.isfinite(response).all()
