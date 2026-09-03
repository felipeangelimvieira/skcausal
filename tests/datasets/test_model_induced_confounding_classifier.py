import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from skcausal.datasets.semi_synthetic_classifier import SemiSyntheticClassifier


def _toy_classification_dataset():
    covariates = pd.DataFrame(
        {
            "age": [18.0, 20.0, 22.0, 30.0, 32.0, 34.0, 42.0, 44.0, 46.0],
            "income": [25.0, 27.0, 29.0, 45.0, 47.0, 49.0, 65.0, 67.0, 69.0],
            "score": [-2.0, -1.5, -1.0, -0.1, 0.0, 0.1, 1.0, 1.5, 2.0],
        }
    )
    labels = pd.Series(
        [
            "basic",
            "basic",
            "basic",
            "standard",
            "standard",
            "standard",
            "premium",
            "premium",
            "premium",
        ],
        name="segment",
    )
    return covariates, labels


def _stringify(values) -> np.ndarray:
    return np.asarray(values, dtype=object).astype(str)


class _HookedSemiSyntheticClassifier(SemiSyntheticClassifier):
    def _load_dataset(self):
        return _toy_classification_dataset()


def test_semisynthetic_classifier_normalizes_dataset_and_uses_predictions_for_treatment():
    dataset = _HookedSemiSyntheticClassifier(
        classifier=LogisticRegression(max_iter=2000),
        random_state=7,
        treatment_effect_scale=0.0,
        outcome_noise_scale=0.0,
    )

    covariates, treatments, outcomes = dataset.load()

    assert isinstance(covariates, pl.DataFrame)
    assert isinstance(treatments, pl.DataFrame)
    assert isinstance(outcomes, pl.DataFrame)
    assert covariates.columns == ["age", "income", "score"]
    assert treatments.columns == ["t"]
    assert outcomes.columns == ["y"]
    assert treatments.schema["t"] == pl.Categorical

    covariate_array = covariates.to_numpy()
    treatment_labels = treatments.get_column("t").cast(pl.Utf8).to_numpy()

    np.testing.assert_allclose(covariate_array.mean(axis=0), np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(
        covariate_array.std(axis=0, ddof=0), np.ones(3), atol=1e-10
    )

    expected_treatments = _stringify(dataset.classifier_.predict(covariate_array))
    np.testing.assert_array_equal(treatment_labels, expected_treatments)
    assert dataset.get_levels().get_column("t").cast(pl.Utf8).to_list() == [
        "basic",
        "premium",
        "standard",
    ]


def test_semisynthetic_classifier_predict_y_accepts_backends_and_matches_probability_plus_effect():
    dataset = SemiSyntheticClassifier(
        classifier=LogisticRegression(max_iter=2000),
        load_dataset=_toy_classification_dataset,
        random_state=3,
        treatment_effect_scale=1.25,
        outcome_noise_scale=0.0,
    )

    covariates, treatments, _ = dataset.load()
    levels = dataset.get_levels()

    predictions_from_polars = dataset.predict_y(covariates, treatments)
    predictions_from_pandas = dataset.predict_y(
        covariates.to_pandas(), treatments.to_pandas()
    )
    predictions_from_numpy = dataset.predict_y(
        covariates.to_numpy(), treatments.to_numpy()
    )
    curve = dataset.predict_curve(covariates, levels)
    legacy_curve = dataset.predict(covariates, levels)

    probabilities = dataset.classifier_.predict_proba(covariates.to_numpy())
    labels = treatments.get_column("t").cast(pl.Utf8).to_numpy()
    class_to_index = {label: idx for idx, label in enumerate(dataset.treatment_levels_)}
    baseline = probabilities[
        np.arange(len(labels)),
        np.asarray([class_to_index[label] for label in labels]),
    ]
    expected = (
        baseline
        + np.asarray(
            [dataset.treatment_effects_[label] for label in labels], dtype=float
        )
    ).reshape(-1, 1)

    np.testing.assert_allclose(predictions_from_polars, expected)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_pandas)
    np.testing.assert_allclose(predictions_from_polars, predictions_from_numpy)
    np.testing.assert_allclose(curve, legacy_curve)
    assert curve.shape == (len(dataset.treatment_levels_),)
    assert levels.schema["t"] == pl.Categorical
    assert levels.get_column("t").cast(pl.Utf8).to_list() == dataset.treatment_levels_
