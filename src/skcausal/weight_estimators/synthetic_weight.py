import copy
from typing import List, Callable
import numpy as np
import polars as pl
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from skcausal.utils.polars import ALL_DTYPES, convert_categorical_to_dummies
from skcausal.weight_estimators.base import BaseBalancingWeightRegressor
from skcausal.utils.sklearn import _resolve_sample_weight_fit_args

__all__ = [
    "SyntheticWeightRegressor",
]


class SyntheticWeightRegressor(BaseBalancingWeightRegressor):
    """Estimate balancing weights via a synthetic classification task.

    The estimator augments the observed data with synthetic observations whose
    treatments are sampled according to ``method``. A classifier is then trained
    to discriminate between real and synthetic samples, and its predicted
    propensities are transformed into balancing weights.

    Parameters
    ----------
    classifier : ClassifierMixin
        Classifier following the scikit-learn API and exposing
        :meth:`predict_proba`.
    treatment_transformation : TransformerMixin, optional
        Transformer applied to the treatment column(s) before concatenation with
        the features. Pass the string ``"spline"`` to use a spline basis with
        10 knots and degree 2. Defaults to ``None``.
    num_synthetic_dataset_generations : float, optional
        Multiplier controlling how many synthetic samples are generated relative
        to the original sample size (``dataset_size = n_samples *
        num_synthetic_dataset_generations``). Defaults to ``1``.
    method : {"uniform", "balanced", "balanced:replace"}, optional
        Strategy used to draw synthetic treatment values. Defaults to
        ``"uniform"``.
    max_trials : int, optional
        Maximum number of attempts when running in convergence or ensemble
        mode. Defaults to ``5``.
    random_state : int, optional
        Seed forwarded to the internal random number generator. Defaults to ``0``.
    fit_mode : {"ensemble", "convergence"}, optional
        Controls how the classifier is trained on the synthetic data. Defaults to
        ``"ensemble"``.
    """

    _tags = {
        "t_inner_mtype": pl.DataFrame,
        "one_hot_encode_enum_columns": False,
        "supported_t_dtypes": ALL_DTYPES,
        "balancing_weight_type": "propensity_score",
    }

    def __init__(
        self,
        classifier: ClassifierMixin,
        treatment_transformation: TransformerMixin = None,
        num_synthetic_dataset_generations: int = 1,
        method: str = "uniform",
        max_trials: int = 5,
        random_state=0,
        fit_mode: str = "ensemble",
        drop_not_converged: bool = False,
        mean_type="arithmetic",
    ):

        self.classifier = classifier
        self.treatment_transformation = treatment_transformation
        self.num_synthetic_dataset_generations = num_synthetic_dataset_generations
        self.method = method
        self.max_trials = max_trials
        self.random_state = random_state
        self.fit_mode = fit_mode
        self.drop_not_converged = drop_not_converged
        self.mean_type = mean_type
        self.classifiers_ = None

        super().__init__()

        self._rng = np.random.default_rng(self.random_state)
        self._treatment_transformation = self.treatment_transformation
        if self.treatment_transformation == "spline":
            self._treatment_transformation = SplineTransformer(n_knots=10, degree=2)

        if method.startswith("balanced"):
            self.set_tags(**{"balancing_weight_type": "stabilized"})

    def _fit(self, X, t):

        self.domain_size_ = domain_size(t)
        if self._treatment_transformation is not None:
            self._treatment_transformation.fit(t.to_numpy().reshape(X.shape[0], -1))

        self.classifier_ = copy.deepcopy(self.classifier)
        self.classifiers_ = None

        if self.fit_mode == "ensemble":
            self._fit_ensemble(X, t)
        elif self.fit_mode == "convergence":
            self._fit_until_convergence(X, t)
        else:
            raise ValueError(f"Invalid fit_mode: {self.fit_mode}")
        return self

    def _fit_until_convergence(self, X, t):

        converged = False
        for trial in range(self.max_trials):
            shuffle_idx = self._rng.permutation(X.shape[0])
            X_classif, y_classif, sample_weights = (
                make_probabilistic_classification_dataset(
                    self._rng,
                    X[shuffle_idx],
                    t[shuffle_idx],
                    method=self.method,
                    dataset_size=int(
                        X.shape[0] * self.num_synthetic_dataset_generations
                    ),
                    feature_transformation=self._make_feature_data,
                )
            )

            fit_kwargs = _resolve_sample_weight_fit_args(
                self.classifier_, sample_weights
            )
            self.classifier_.fit(X_classif, y_classif, **fit_kwargs)

            probas = self.classifier_.predict_proba(X_classif)

            if probas[:, 0].std() > 1e-6:
                converged = True

            if converged:
                break

        if not converged:
            raise ValueError(
                "Standard deviation of weights is too low" + "Check your model."
            )

    def _fit_ensemble(self, X, t):

        self.classifiers_ = []
        for trial in range(self.max_trials):
            shuffle_idx = self._rng.permutation(X.shape[0])
            X_classif, y_classif, sample_weights = (
                make_probabilistic_classification_dataset(
                    self._rng,
                    X[shuffle_idx],
                    t[shuffle_idx],
                    method=self.method,
                    dataset_size=int(
                        X.shape[0] * self.num_synthetic_dataset_generations
                    ),
                    feature_transformation=self._make_feature_data,
                )
            )

            fit_kwargs = _resolve_sample_weight_fit_args(
                self.classifier_, sample_weights
            )

            _classif = copy.deepcopy(self.classifier_).fit(
                X_classif, y_classif, **fit_kwargs
            )
            probas = _classif.predict_proba(X_classif)

            converged = probas[:, 0].std() > 1e-6

            if self.drop_not_converged and not converged:
                print(f"Trial {trial} did not converge, skipping.")
                continue

            self.classifiers_.append(_classif)
        if not self.classifiers_:
            raise ValueError(
                "No ensemble members converged; consider relaxing constraints."
            )

        # Ensemble predictions are computed via _predict_with_classifiers.
        self.classifier_ = None

    def _predict_sample_weight(self, X, t):

        t = convert_categorical_to_dummies(t)
        t = t.to_numpy().astype(float)

        Xt = self._make_feature_data(X, t)

        batch_size = 100_000
        n_samples = Xt.shape[0]

        if n_samples <= batch_size:
            probas = self._predict_with_classifiers(Xt)
        else:
            probas = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_Xt = Xt[start:end]
                batch_probas = self._predict_with_classifiers(batch_Xt)
                probas.append(batch_probas)

            probas = np.vstack(probas)

        w = probas[:, 1] / (probas[:, 0] + 1e-9)
        w *= self.domain_size_
        return w.reshape(-1, 1)

    def _make_feature_data(self, X, t):
        if self._treatment_transformation is not None:
            t = self._treatment_transformation.fit_transform(t)
        Xt = np.concatenate([X, t], axis=1)
        return Xt

    def _predict_with_classifiers(self, Xt):
        """Run the trained classifier(s) on Xt."""

        if self.classifiers_:
            return self._aggregate_classifier_predictions(Xt)

        if self.classifier_ is None:
            raise ValueError(
                "SyntheticWeightRegressor must be fitted before predicting."
            )

        return self.classifier_.predict_proba(Xt)

    def _aggregate_classifier_predictions(self, Xt):
        probas = [clf.predict_proba(Xt) for clf in self.classifiers_]

        if not probas:
            raise ValueError("No classifiers available for aggregation.")

        probas = np.stack(probas, axis=0)
        if self.mean_type == "geometric":
            # Avoid division by zero by adding small epsilon? not necessary maybe.
            return np.prod(probas, axis=0) ** (1 / probas.shape[0])
        if self.mean_type == "arithmetic":
            return np.mean(probas, axis=0)

        raise ValueError(
            f"Unsupported mean_type '{self.mean_type}'. Use 'arithmetic' or 'geometric'."
        )

    @classmethod
    def get_test_params(self):
        from sklearn.ensemble import RandomTreesEmbedding
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        random_tree_embedding = RandomTreesEmbedding(
            n_estimators=5, max_depth=2, random_state=0
        )
        rt_model = make_pipeline(
            random_tree_embedding, LogisticRegression(max_iter=1000)
        )

        return [
            {
                "classifier": LogisticRegression(),
                "num_synthetic_dataset_generations": 2,
                "method": "uniform",
            },
            {
                "classifier": LogisticRegression(),
                "num_synthetic_dataset_generations": 2,
                "method": "balanced",
            },
            {
                "classifier": rt_model,
                "num_synthetic_dataset_generations": 2,
                "method": "balanced",
            },
        ]


def make_probabilistic_classification_dataset(
    rng: np.random.Generator,
    X: pl.DataFrame,
    t: pl.DataFrame,
    method: str,
    dataset_size: int,
    feature_transformation: Callable = lambda X, t: np.concatenate([X, t], axis=1),
):
    """Construct the real/synthetic classification dataset.

    Parameters
    ----------
    X : pl.DataFrame
        Covariate matrix containing the original observations.
    t : pl.DataFrame
        Observed treatment values aligned with ``X``.
    method : str
        Sampling strategy forwarded to :func:`sample_synthetic_random`.
    dataset_size : int
        Number of synthetic treatment draws to generate.

    Returns
    -------
    np.ndarray
        Concatenated covariate/treatment features for the classification
        task.
    np.ndarray
        Labels indicating whether each row is synthetic (1) or original
        (0).
    """

    X_target, t_target = sample_synthetic_random(
        rng,
        X=X,
        t=t,
        method=method,
        dataset_size=dataset_size,
    )

    t_target = convert_categorical_to_dummies(t_target)
    t_target = t_target.to_numpy().astype(float)

    t = convert_categorical_to_dummies(t)
    t = t.to_numpy().astype(float)

    Xt_target = feature_transformation(X_target, t_target)
    Xt = feature_transformation(X, t)

    Xt_all = np.concatenate([Xt_target, Xt], axis=0)
    y = np.concatenate([np.ones(Xt_target.shape[0]), np.zeros(Xt.shape[0])])

    sample_weights = np.ones_like(y)
    sample_weights[y == 0] = 1 / (y == 0).sum() * sample_weights.shape[0] / 2
    sample_weights[y == 1] = 1 / (y == 1).sum() * sample_weights.shape[0] / 2

    return Xt_all.astype(float), y.astype(float), sample_weights.astype(float)


def sample_synthetic_random(
    rng: np.random.Generator,
    X: pl.DataFrame,
    t: pl.DataFrame,
    method: str,
    dataset_size: int,
):
    """Sample synthetic treatment values according to the chosen strategy.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator used for sampling.
    X : pl.DataFrame
        Original feature matrix used to build the synthetic dataset.
    t : pl.DataFrame
        Observed treatment values.
    method : str
        Sampling mode. Supported values are ``"balanced"``,
        ``"balanced:replace"``, and ``"uniform"``.
    dataset_size : int
        Number of synthetic samples to create.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Synthetic covariates and treatments of size ``dataset_size``.
    """
    idx = np.arange(dataset_size) % len(X)
    _X = X[idx]
    if method == "balanced:replace":
        _t = t[rng.choice(len(t), size=dataset_size, replace=True)]
    elif method == "balanced":
        n_samplings = dataset_size // len(t)

        mem = []
        for i in range(n_samplings):
            _t = t[rng.choice(len(t), size=len(X), replace=False)]
            mem.append(_t)
        _t = pl.concat(mem)

    elif method == "uniform":
        sampling_func = uniform_treatment_sampler_factory(rng, t.schema)
        _t = sampling_func(dataset_size)
    else:
        raise ValueError(f"Invalid method: {method}")
    return _X, _t


def uniform_treatment_sampler_factory(rng, t_schema: pl.Schema):

    samplers = []
    for dtype in t_schema.dtypes():
        if dtype == pl.Enum:
            _sampler = _random_category(dtype.categories)
            samplers.append(_sampler)
        elif dtype == pl.Binary or dtype == pl.Boolean:
            samplers.append(_random_category([0, 1]))
        elif dtype in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            raise ValueError("Cannot sample from UInt. Use Enum or Binary")
        elif dtype.is_numeric():
            samplers.append(_random_uniform_continuous())
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

    sampler = _strategy_sampler(rng, samplers, t_schema)
    return sampler


def domain_size(t):
    size = 1
    schema = t.schema
    for column in schema.names():
        dtype = schema[column]
        if dtype == pl.Enum:
            size *= len(dtype.categories)
        elif dtype == pl.Binary or dtype == pl.Boolean:
            size *= 2
        elif dtype in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            raise ValueError("Cannot compute domain size from UInt. Use Enum or Binary")
        elif dtype.is_numeric():
            size = t[column].max() - t[column].min()
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
    return size


def _random_category(categories):
    def _sample_category(rng, n_samples):
        return rng.choice(categories, n_samples)

    return _sample_category


def _random_uniform_continuous():
    def _sample_continuous(rng, n_samples):
        return rng.uniform(0, 1, n_samples)

    return _sample_continuous


def _strategy_sampler(
    rng, sampling_strategies: List[Callable], t_schema: pl.Schema
) -> pl.DataFrame:

    assert len(sampling_strategies) == len(t_schema)

    def func(n_samples):
        data = {}
        for strategy, column_name in zip(sampling_strategies, t_schema.names()):
            data[column_name] = strategy(rng, n_samples)
        return pl.DataFrame(data, schema=t_schema)

    return func
