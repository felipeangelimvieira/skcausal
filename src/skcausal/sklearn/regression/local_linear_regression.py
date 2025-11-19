import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LocalLinearRegression(BaseEstimator, RegressorMixin):
    """A local linear regression model compliant with scikit-learn's API.

    Parameters
    ----------
    length_scale : float, default=1.0
        The length_scale parameter for the kernel.
    kernel : {'gaussian', 'epanechnikov'}, default='gaussian'
        The type of kernel to use for weighting.
    """

    def __init__(self, length_scale=1.0, kernel="gaussian"):
        self.length_scale = length_scale
        self.kernel = kernel

        if not isinstance(self.length_scale, (int, float)):
            raise ValueError("length_scale must be a number.")
        if kernel not in ["gaussian", "epanechnikov"]:
            raise ValueError("Unsupported kernel.")

    def _kernel(self, u):
        """Compute kernel weights based on scaled distances u."""
        if self.kernel == "gaussian":
            return np.exp(-0.5 * (u**2))
        elif self.kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit(self, X, y):
        """Store the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """Predict using the local linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data for prediction.
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"Expected {self.X_.shape[1]} features, got {X.shape[1]}")

        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i, :]
            differences = self.X_ - x
            distances = np.linalg.norm(differences, axis=1)
            u = distances / self.length_scale
            weights = self._kernel(u)

            if np.sum(weights) == 0:
                y_pred[i] = np.mean(self.y_)
                continue

            # Construct design matrix with intercept and centered features
            X_design = np.hstack([np.ones((self.X_.shape[0], 1)), differences])
            XW = X_design * weights[:, np.newaxis]  # Weight each row
            XWX = XW.T @ X_design
            XWy = XW.T @ self.y_

            try:
                beta = np.linalg.solve(XWX, XWy)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(XWX, XWy, rcond=None)[0]

            y_pred[i] = beta[0]

        return y_pred
