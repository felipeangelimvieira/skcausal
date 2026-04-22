import numpy as np
import pandas as pd
import polars as pl
import pytest

from skcausal.density.sklearn import SklearnCategoricalDensity

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _make_multiclass_dataset(n_samples=400, random_state=42):
    rng = np.random.default_rng(random_state)

    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    logits = np.column_stack(
        [
            1.2 * x0 - 0.3 * x1,
            -0.8 * x0 + 0.5 * x1,
            0.2 * x0 - 0.1 * x1,
        ]
    )
    logits += rng.normal(scale=0.2, size=logits.shape)
    labels = np.asarray(["control", "treated", "placebo"])[logits.argmax(axis=1)]

    X = pd.DataFrame({"x0": x0, "x1": x1})
    t = pd.DataFrame({"t": labels})
    return X, t


def _make_binary_dataset(n_samples=400, random_state=42):
    rng = np.random.default_rng(random_state)

    x0 = rng.normal(size=n_samples)
    x1 = rng.normal(size=n_samples)
    logits = 1.0 * x0 - 0.7 * x1 + rng.normal(scale=0.2, size=n_samples)
    labels = logits > 0.0

    X = pd.DataFrame({"x0": x0, "x1": x1})
    t = pd.DataFrame({"t": labels})
    return X, t


def _observed_class_probability(classifier, X, t):
    probabilities = classifier.predict_proba(X)
    labels = t["t"].to_numpy()
    class_to_index = {label: idx for idx, label in enumerate(classifier.classes_)}
    return probabilities[
        np.arange(len(labels)),
        np.asarray([class_to_index[label] for label in labels]),
    ].reshape(-1, 1)


def test_sklearn_categorical_density_matches_direct_classifier_probability_binary():
    X, t = _make_binary_dataset()
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        t,
        test_size=0.3,
        random_state=0,
        stratify=t["t"],
    )

    direct_classifier = LogisticRegression(max_iter=1000)
    wrapped_estimator = SklearnCategoricalDensity(
        classifier=LogisticRegression(max_iter=1000)
    )

    direct_classifier.fit(X_train, t_train["t"])
    wrapped_estimator.fit(pl.from_pandas(X_train), pl.from_pandas(t_train))

    direct_density = _observed_class_probability(direct_classifier, X_test, t_test)
    wrapped_density = wrapped_estimator.predict_density(
        pl.from_pandas(X_test),
        pl.from_pandas(t_test),
    )

    assert np.allclose(wrapped_density, direct_density)
    assert wrapped_density.shape == (len(X_test), 1)


def test_sklearn_categorical_density_matches_direct_classifier_probability_multiclass():
    X, t = _make_multiclass_dataset()
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        t,
        test_size=0.3,
        random_state=0,
        stratify=t["t"],
    )

    direct_classifier = LogisticRegression(max_iter=1000)
    wrapped_estimator = SklearnCategoricalDensity(
        classifier=LogisticRegression(max_iter=1000)
    )

    direct_classifier.fit(X_train, t_train["t"])
    wrapped_estimator.fit(pl.from_pandas(X_train), pl.from_pandas(t_train))

    direct_density = _observed_class_probability(direct_classifier, X_test, t_test)
    wrapped_density = wrapped_estimator.predict_density(
        pl.from_pandas(X_test),
        pl.from_pandas(t_test),
    )

    assert np.allclose(wrapped_density, direct_density)
    assert wrapped_density.shape == (len(X_test), 1)


class _PositiveClassOnlyBinaryClassifier:
    def __init__(self, return_column_vector=False):
        self.return_column_vector = return_column_vector

    def fit(self, X, y):
        self.estimator_ = LogisticRegression(max_iter=1000)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        positive_probability = self.estimator_.predict_proba(X)[:, 1]
        if self.return_column_vector:
            return positive_probability.reshape(-1, 1)
        return positive_probability


@pytest.mark.parametrize("return_column_vector", [False, True])
def test_sklearn_categorical_density_accepts_binary_positive_class_outputs(
    return_column_vector,
):
    X, t = _make_binary_dataset()
    X_train, X_test, t_train, t_test = train_test_split(
        X,
        t,
        test_size=0.3,
        random_state=0,
        stratify=t["t"],
    )

    reference_classifier = LogisticRegression(max_iter=1000)
    reference_classifier.fit(X_train, t_train["t"])

    wrapped_estimator = SklearnCategoricalDensity(
        classifier=_PositiveClassOnlyBinaryClassifier(
            return_column_vector=return_column_vector
        )
    )
    wrapped_estimator.fit(pl.from_pandas(X_train), pl.from_pandas(t_train))

    expected_density = _observed_class_probability(
        reference_classifier,
        X_test,
        t_test,
    )
    wrapped_density = wrapped_estimator.predict_density(
        pl.from_pandas(X_test),
        pl.from_pandas(t_test),
    )

    assert np.allclose(wrapped_density, expected_density)


def test_sklearn_categorical_density_raises_on_unseen_label():
    X_train = pl.DataFrame({"x0": [0.0, 1.0, 2.0, 3.0], "x1": [0.0, 1.0, 0.0, 1.0]})
    t_train = pl.DataFrame({"t": ["control", "treated", "control", "treated"]})
    X_test = pl.DataFrame({"x0": [0.5], "x1": [0.5]})
    t_test = pl.DataFrame({"t": ["missing"]})

    estimator = SklearnCategoricalDensity(LogisticRegression(max_iter=1000))
    estimator.fit(X_train, t_train)

    with pytest.raises(ValueError, match="Observed treatment label was not seen"):
        estimator.predict_density(X_test, t_test)
