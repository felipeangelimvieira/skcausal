import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class AveragingClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom classifier that takes a list of already-fitted
    classifiers and returns an ensemble prediction by averaging
    their predicted probabilities.
    """

    def __init__(self, classifiers, mean="arithmetic"):
        """
        Parameters
        ----------
        classifiers : list
            A list of already-fitted sklearn classifiers.
        mean : str
            The type of mean to use for averaging predictions.
            Supported values are "arithmetic" and "geometric".
        """
        self.classifiers = classifiers
        self.mean = mean
        super().__init__()

    def fit(self, X, y=None):
        """
        This method doesn't do anything because the classifiers
        are already fitted. It just returns self.
        """
        return self

    def predict_proba(self, X):
        """
        Compute the average predicted probabilities over all classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg_probas : ndarray of shape (n_samples, n_classes)
            The averaged predicted probabilities.
        """
        if self.mean == "geometric":
            # Collect predict_proba from each classifier
            probas = [clf.predict_proba(X) for clf in self.classifiers]
            # Multiply the arrays element-wise
            prod_probas = np.prod(probas, axis=0)
            # Take the nth root to get the geometric mean
            geo_mean_probas = prod_probas ** (1 / len(self.classifiers))
            return geo_mean_probas
        # Collect predict_proba from each classifier
        probas = [clf.predict_proba(X) for clf in self.classifiers]
        # Sum the arrays along axis=0 (element-wise)
        sum_probas = np.sum(probas, axis=0)
        # Divide by the number of classifiers to get the average
        avg_probas = sum_probas / len(self.classifiers)
        return avg_probas

    def predict(self, X):
        """
        Predict class labels for the given samples using
        the averaged probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        avg_probas = self.predict_proba(X)
        # Class with the highest probability
        return np.argmax(avg_probas, axis=1)
