import numpy as np
from sklearn.linear_model import LogisticRegression

from skcausal.density.cbps._design import DesignStandardizer
from skcausal.density.cbps._glm import fit_linear, fit_logistic, fit_multinomial


def _make_design(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return X, DesignStandardizer().fit(X).u_, rng


def test_logistic_matches_sklearn_unpenalized():
    X, U, rng = _make_design()
    y = (X @ [1.0, -0.5, 0.2] + rng.normal(size=len(X)) > 0).astype(float)

    beta = fit_logistic(U, y)

    reference = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-10)
    reference.fit(U, y)
    np.testing.assert_allclose(beta, reference.coef_.ravel(), atol=1e-4)


def test_multinomial_with_two_levels_equals_logistic():
    X, U, rng = _make_design()
    y = (X @ [1.0, -0.5, 0.2] + rng.normal(size=len(X)) > 0).astype(float)
    T = np.column_stack([1 - y, y])

    beta = fit_multinomial(U, T)

    assert beta.shape == (U.shape[1], 1)
    np.testing.assert_allclose(beta[:, 0], fit_logistic(U, y), atol=1e-5)


def test_multinomial_matches_sklearn_three_levels():
    X, U, rng = _make_design()
    logits = np.column_stack(
        [np.zeros(len(X)), X @ [1.0, 0.0, 0.5], X @ [-0.5, 1.0, 0.0]]
    )
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(3, p=p) for p in probs])
    T = np.eye(3)[labels]

    beta = fit_multinomial(U, T)

    reference = LogisticRegression(penalty=None, fit_intercept=False, tol=1e-10)
    reference.fit(U, labels)
    # sklearn uses symmetric parametrization; compare log-odds vs baseline.
    expected = (reference.coef_[1:] - reference.coef_[0]).T
    np.testing.assert_allclose(beta, expected, atol=1e-3)


def test_linear_equals_lstsq():
    X, U, rng = _make_design()
    y = X @ [1.0, -0.5, 0.2] + rng.normal(size=len(X))

    np.testing.assert_allclose(fit_linear(U, y), np.linalg.lstsq(U, y, rcond=None)[0])
