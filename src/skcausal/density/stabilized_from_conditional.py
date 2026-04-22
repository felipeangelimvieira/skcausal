"""Wrappers that turn a conditional density into a stabilized density ratio."""

import copy

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from skcausal.density.base import BaseDensityEstimator

__all__ = ["KernelMarginalAndConditional", "IntegratedMarginalAndConditional"]


def _validate_conditional_density_estimator(estimator: BaseDensityEstimator) -> None:
    """Validate that a wrapped estimator represents ``p(t | x)``.

    Both stabilized wrappers assume the supplied estimator returns a proper
    conditional treatment density. This helper enforces that contract in one
    place by checking the base type first and then verifying that the declared
    ``density_kind`` tag is ``"conditional"``. Wrapping an estimator that
    already returns a stabilized ratio would apply the marginal correction a
    second time and therefore change the target quantity.

    Parameters
    ----------
    estimator : BaseDensityEstimator
        Estimator candidate that will be wrapped by a stabilized-density
        adapter.

    Returns
    -------
    None
        This function returns nothing. It raises an exception when the wrapped
        estimator does not satisfy the required interface or semantics.
    """
    if not isinstance(estimator, BaseDensityEstimator):
        raise TypeError(
            "conditional_density_estimator must be an instance of "
            "BaseDensityEstimator."
        )

    density_kind = estimator.get_tag("density_kind", "conditional")
    if density_kind != "conditional":
        raise ValueError(
            "conditional_density_estimator must predict a conditional density, "
            f"but received density_kind={density_kind!r}."
        )


class KernelMarginalAndConditional(BaseDensityEstimator):
    """
    Use a kernel to estimate the marginal P(T) and an estimator for P(T|X).

    Uses conditional_density_estimator to estimate P(T|X) and KDE to estimate
    P(T), returning P(T|X) / P(T).

    In the case of categorical dimensions, we compute P(T_{cat}) for each
    combination of categorical values, and then fit a copy of the kernel on
    P(T_{cont} | T_{cat}) for each combination of categorical values.
    The final estimate for P(T) is P(T_{cat}) * P(T_{cont} | T_{cat}).

    Parameters
    ----------
    conditional_density_estimator: BaseDensityEstimator
        A conditional density estimator (instance of
        BaseDensityEstimator)
    kernel: KernelDensity
        A sklearn kernel density estimator
    """

    _tags = {
        "backend": "pandas",
        "density_kind": "stabilized",
    }

    def __init__(
        self, conditional_density_estimator: BaseDensityEstimator, kernel: KernelDensity
    ):
        """Store the wrapped conditional estimator and KDE template.

        The passed ``kernel`` is treated as an unfitted template. During
        :meth:`_fit`, the estimator either clones it once for a purely
        continuous marginal model or clones it once per observed categorical
        treatment combination for the mixed-treatment case. The wrapped
        conditional estimator is also cloned so fitting this wrapper does not
        mutate the user-supplied estimator instance.

        Parameters
        ----------
        conditional_density_estimator : BaseDensityEstimator
            Estimator used to model the conditional treatment density
            ``p(t | x)``.
        kernel : KernelDensity
            Unfitted KDE template used for the marginal treatment model.

        Returns
        -------
        None
            Initializes estimator state and validates the wrapped estimator.
        """
        self.conditional_density_estimator = conditional_density_estimator
        self.kernel = kernel
        super().__init__()

        _validate_conditional_density_estimator(self.conditional_density_estimator)

        self.set_tags(
            **{
                "capability:t_type": self.conditional_density_estimator.get_tag(
                    "capability:t_type",
                    ["continuous", "categorical"],
                ),
                "capability:multidimensional_treatment": (
                    self.conditional_density_estimator.get_tag(
                        "capability:multidimensional_treatment",
                        False,
                    )
                ),
            }
        )

        self.conditional_density_estimator_ = self.conditional_density_estimator.clone()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        """Fit the marginal model for ``p(t)`` and the wrapped model for ``p(t|x)``.

        The stabilized ratio requires both a marginal treatment density and a
        conditional treatment density. This method first fits the marginal side
        using only the observed treatment sample ``t``. It then fits the cloned
        wrapped estimator on ``(X, t)`` so later prediction can divide the two
        quantities row-wise.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix passed to the wrapped conditional estimator.
        t : pd.DataFrame
            Observed treatment sample used to fit both the marginal and
            conditional models.

        Returns
        -------
        KernelMarginalAndConditional
            Fitted estimator instance.
        """
        self._fit_marginal_density_model(t)
        self.conditional_density_estimator_.fit(X, t)
        return self

    def _predict_density(self, X, t):
        """Evaluate the stabilized density ratio for the supplied rows.

        Prediction proceeds in two steps. First, the fitted marginal model is
        evaluated at each treatment row to produce ``p(t)``. Next, the wrapped
        conditional estimator is evaluated on the same rows to produce
        ``p(t | x)``. The returned value is their ratio, with the marginal term
        clipped away from zero for numerical stability.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate values at which the conditional density is evaluated.
        t : pd.DataFrame
            Treatment values at which both marginal and conditional densities
            are evaluated.

        Returns
        -------
        np.ndarray
            Column vector whose ``i``-th entry is the stabilized ratio
            ``p(t_i | x_i) / p(t_i)``.
        """
        pt = self._predict_marginal_density(t)
        pt_given_x = self.conditional_density_estimator_.predict_density(X, t)
        stabilized_density = pt_given_x / np.clip(pt, np.finfo(float).tiny, None)
        return stabilized_density

    def _fit_marginal_density_model(self, t: pd.DataFrame):
        """Fit the marginal treatment model for the observed treatment sample.

        The fit-time treatment metadata is used to split treatment columns into
        continuous and categorical groups. If categorical columns are present,
        the method stores empirical joint masses for each observed categorical
        combination and, when continuous columns also exist, fits one cloned KDE
        to the continuous slice inside each categorical group. If treatment is
        purely continuous, it fits a single global KDE instead.

        Parameters
        ----------
        t : pd.DataFrame
            Treatment sample from the training data.

        Returns
        -------
        None
            Stores fitted marginal-model components on the estimator.
        """
        self.t_column_types_ = self._t_metadata["t_column_types"]
        self.continuous_t_columns_ = [
            column
            for column, column_type in self.t_column_types_.items()
            if column_type == "continuous"
        ]
        self.categorical_t_columns_ = [
            column
            for column, column_type in self.t_column_types_.items()
            if column_type == "categorical"
        ]

        self.global_continuous_kernel_ = None
        self.category_probabilities_ = {}
        self.category_kernels_ = {}

        if self.categorical_t_columns_:
            total_count = float(len(t))
            for category_key, group in t.groupby(
                self.categorical_t_columns_,
                observed=True,
                sort=False,
            ):
                normalized_key = self._normalize_category_key(category_key)
                self.category_probabilities_[normalized_key] = len(group) / total_count

                if self.continuous_t_columns_:
                    self.category_kernels_[normalized_key] = self._fit_kernel(
                        group[self.continuous_t_columns_]
                    )
            return

        if self.continuous_t_columns_:
            self.global_continuous_kernel_ = self._fit_kernel(
                t[self.continuous_t_columns_]
            )

    def _predict_marginal_density(self, t: pd.DataFrame) -> np.ndarray:
        """Evaluate the fitted marginal treatment density at the supplied rows.

        The marginal model is assembled multiplicatively. For mixed treatment,
        the method first multiplies in the empirical mass of the categorical
        treatment combination and then multiplies in the continuous density from
        the corresponding group-specific KDE. For purely continuous treatment it
        scores a single global KDE, and for purely categorical treatment it
        returns only the empirical combination masses.

        Parameters
        ----------
        t : pd.DataFrame
            Treatment rows at which to evaluate the fitted marginal model.

        Returns
        -------
        np.ndarray
            Column vector of marginal treatment densities ``p(t)``.
        """
        marginal_density = np.ones((len(t), 1), dtype=float)
        category_keys = None

        if self.categorical_t_columns_:
            category_keys = self._get_category_keys(t)
            marginal_density *= self._predict_category_mass(category_keys)

        if self.continuous_t_columns_:
            if self.categorical_t_columns_:
                marginal_density *= self._predict_groupwise_continuous_density(
                    t,
                    category_keys,
                )
            else:
                marginal_density *= self._score_kernel(
                    self.global_continuous_kernel_,
                    t[self.continuous_t_columns_],
                )

        return marginal_density

    def _predict_category_mass(self, category_keys: list[tuple]) -> np.ndarray:
        """Look up empirical masses for categorical treatment combinations.

        Each key represents one normalized joint assignment of the categorical
        treatment columns. The returned masses are the empirical frequencies
        estimated at fit time. Keys that were not observed during fitting are
        assigned zero mass.

        Parameters
        ----------
        category_keys : list[tuple]
            Canonical categorical treatment keys, one per prediction row.

        Returns
        -------
        np.ndarray
            Column vector of empirical probabilities for the supplied
            categorical treatment combinations.
        """
        category_mass = np.array(
            [self.category_probabilities_.get(key, 0.0) for key in category_keys],
            dtype=float,
        )
        return category_mass.reshape(-1, 1)

    def _predict_groupwise_continuous_density(
        self,
        t: pd.DataFrame,
        category_keys: list[tuple],
    ) -> np.ndarray:
        """Score continuous treatment columns with the matching per-category KDE.

        Rows are first bucketed by their normalized categorical treatment key.
        Each bucket is then scored with the KDE fitted for that categorical
        combination during :meth:`_fit_marginal_density_model`. If a category is
        unseen at prediction time, its density contribution remains zero because
        there is no fitted conditional KDE for that group.

        Parameters
        ----------
        t : pd.DataFrame
            Treatment rows whose continuous columns should be scored.
        category_keys : list[tuple]
            Canonical categorical keys corresponding to the rows of ``t``.

        Returns
        -------
        np.ndarray
            Column vector of conditional continuous densities
            ``p(t_cont | t_cat)``.
        """
        continuous_density = np.zeros((len(t), 1), dtype=float)
        row_indices_by_key = {}
        for row_index, category_key in enumerate(category_keys):
            row_indices_by_key.setdefault(category_key, []).append(row_index)

        for category_key, row_indices in row_indices_by_key.items():
            kernel = self.category_kernels_.get(category_key)
            if kernel is None:
                continue

            continuous_density[np.asarray(row_indices), :] = self._score_kernel(
                kernel,
                t.iloc[row_indices][self.continuous_t_columns_],
            )

        return continuous_density

    def _fit_kernel(self, t_continuous: pd.DataFrame) -> KernelDensity:
        """Clone and fit the configured KDE on continuous treatment columns.

        The clone step keeps the user-supplied ``kernel`` template untouched and
        avoids sharing fitted state across groups. The fitted kernel is always
        trained on a dense floating-point array extracted from the continuous
        treatment columns.

        Parameters
        ----------
        t_continuous : pd.DataFrame
            Continuous treatment columns for one marginal-density fit context,
            either the full sample or a single categorical subgroup.

        Returns
        -------
        KernelDensity
            Fitted KDE instance.
        """
        kernel = copy.deepcopy(self.kernel)
        kernel.fit(t_continuous.to_numpy(dtype=float))
        return kernel

    @staticmethod
    def _score_kernel(kernel: KernelDensity, t_continuous: pd.DataFrame) -> np.ndarray:
        """Evaluate a fitted KDE and return densities as a column vector.

        ``KernelDensity.score_samples`` returns log densities, so this helper
        exponentiates the result and reshapes it into the column-vector format
        expected by the density API.

        Parameters
        ----------
        kernel : KernelDensity
            Previously fitted KDE.
        t_continuous : pd.DataFrame
            Continuous treatment rows to evaluate.

        Returns
        -------
        np.ndarray
            Column vector of density values produced by the KDE.
        """
        log_density = kernel.score_samples(t_continuous.to_numpy(dtype=float))
        return np.exp(log_density).reshape(-1, 1)

    def _get_category_keys(self, t: pd.DataFrame) -> list[tuple]:
        """Extract normalized categorical treatment keys from each row.

        The categorical columns are read row-wise and converted to a canonical
        tuple representation so that dictionary lookups at prediction time match
        the keys produced during fit-time grouping.

        Parameters
        ----------
        t : pd.DataFrame
            Treatment rows whose categorical columns should be converted into
            dictionary keys.

        Returns
        -------
        list[tuple]
            One normalized categorical key per row.
        """
        categorical_frame = t[self.categorical_t_columns_]
        return [
            self._normalize_category_key(category_key)
            for category_key in categorical_frame.itertuples(index=False, name=None)
        ]

    @classmethod
    def _normalize_category_key(cls, category_key) -> tuple:
        """Canonicalize a categorical group key for stable dictionary lookup.

        Pandas may represent a one-column group key as a scalar and a
        multi-column key as a tuple. It may also wrap scalar values in numpy or
        pandas scalar types. This helper forces every key into the same final
        representation: a tuple of normalized Python scalars.

        Parameters
        ----------
        category_key : object
            Group key produced by pandas grouping or row-wise iteration.

        Returns
        -------
        tuple
            Canonical key suitable for use in ``category_probabilities_`` and
            ``category_kernels_`` dictionaries.
        """
        if not isinstance(category_key, tuple):
            category_key = (category_key,)

        return tuple(cls._normalize_scalar(value) for value in category_key)

    @staticmethod
    def _normalize_scalar(value):
        """Unwrap scalar wrapper types while leaving Python values unchanged.

        Some categorical values arrive as numpy or pandas scalar wrappers whose
        equality and hashing behavior can differ from plain Python scalars in
        subtle ways. This helper collapses those wrappers through ``item()``
        when possible so normalized category keys compare reliably.

        Parameters
        ----------
        value : object
            Scalar value drawn from a categorical treatment column.

        Returns
        -------
        object
            Plain Python scalar when conversion is possible, otherwise the
            original value.
        """
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                return value
        return value

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return estimator configurations used by the object test suite.

        The default configuration wraps the naive conditional density estimator
        with a KDE-based marginal model so generic object tests can instantiate
        the class without additional setup.

        Parameters
        ----------
        parameter_set : str, default="default"
            Selector required by the skbase test interface. Only the default set
            is currently implemented.

        Returns
        -------
        list[dict]
            List of constructor argument dictionaries for test instances.
        """
        from skcausal.density.naive import NaiveDensityEstimator
        from sklearn.neighbors import KernelDensity

        return [
            {
                "conditional_density_estimator": NaiveDensityEstimator(),
                "kernel": KernelDensity(bandwidth="scott"),
            }
        ]


class IntegratedMarginalAndConditional(BaseDensityEstimator):
    """
    Use a cond. density estimator to estimate P(T|X) and marginal P(T).

    Uses conditional_density_estimator to estimate P(T|X) and integrates P(T|X)
    over the empirical distribution of X to get P(T),
    returning the ratio P(T|X) / P(T).

    Parameters
    ----------
    conditional_density_estimator: BaseDensityEstimator
        A conditional density estimator (instance of
        BaseDensityEstimator)
    n_samples : int
        Number of samples to use when integrating over the empirical distribution
        of X to get P(T). Default is 1000.
    max_batch_size : int
        Maximum batch size (# of t values to marginalize together)
        when computing the marginal density to avoid memory issues.
        Default is 256.
    """

    _tags = {
        "backend": "pandas",
        "density_kind": "stabilized",
    }

    def __init__(
        self,
        conditional_density_estimator: BaseDensityEstimator,
        n_samples: int = 1000,
        max_batch_size: int = 256,
    ):
        """Store the wrapped estimator and empirical-integration controls.

        This wrapper estimates ``p(t)`` by averaging the wrapped conditional
        density over sampled covariate rows from the training data. The
        ``n_samples`` and ``max_batch_size`` arguments control that Monte Carlo
        approximation and its memory footprint.

        Parameters
        ----------
        conditional_density_estimator : BaseDensityEstimator
            Estimator used to evaluate the conditional treatment density
            ``p(t | x)``.
        n_samples : int, default=1000
            Number of training covariate rows to use when approximating the
            marginal treatment density.
        max_batch_size : int, default=256
            Maximum number of treatment rows to marginalize in a single batch.

        Returns
        -------
        None
            Initializes estimator state and validates the integration controls.
        """
        self.conditional_density_estimator = conditional_density_estimator
        self.n_samples = n_samples
        self.max_batch_size = max_batch_size
        super().__init__()

        _validate_conditional_density_estimator(self.conditional_density_estimator)
        if not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer.")

        self.set_tags(
            **{
                "capability:t_type": self.conditional_density_estimator.get_tag(
                    "capability:t_type",
                    ["continuous", "categorical"],
                ),
                "capability:multidimensional_treatment": (
                    self.conditional_density_estimator.get_tag(
                        "capability:multidimensional_treatment",
                        False,
                    )
                ),
            }
        )

        self.conditional_density_estimator_ = self.conditional_density_estimator.clone()

    def _fit(self, X: pd.DataFrame, t: pd.DataFrame):
        """Fit the wrapped estimator and cache rows for empirical integration.

        The wrapped conditional estimator is fitted on the full training data.
        Afterwards, a subset of the observed covariate rows is stored so that
        :meth:`_estimate_marginal_density` can approximate the expectation
        ``p(t) = E_X[p(t | X)]`` without revisiting the full training matrix on
        every prediction call.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate matrix from the training sample.
        t : pd.DataFrame
            Treatment matrix from the training sample.

        Returns
        -------
        IntegratedMarginalAndConditional
            Fitted estimator instance.
        """
        self.conditional_density_estimator_.fit(X, t)
        self.integration_X_ = self._select_integration_rows(X)
        self.n_integration_rows_ = len(self.integration_X_)
        return self

    def _predict_density(self, X: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
        """Return the stabilized density ratio using an empirical ``p(t)``.

        The wrapped estimator provides the conditional density ``p(t | x)``.
        The marginal term ``p(t)`` is approximated by integrating that same
        conditional density over cached training covariate rows. The final ratio
        divides the two terms row-wise with clipping to avoid numerical issues
        when the estimated marginal density is extremely small.

        Parameters
        ----------
        X : pd.DataFrame
            Covariate values at which to evaluate the conditional density.
        t : pd.DataFrame
            Treatment values at which to evaluate both the conditional and
            marginal densities.

        Returns
        -------
        np.ndarray
            Column vector of stabilized ratios ``p(t_i | x_i) / p(t_i)``.
        """
        pt_given_x = self.conditional_density_estimator_.predict_density(X, t)
        pt = self._estimate_marginal_density(t)
        return pt_given_x / np.clip(pt, np.finfo(float).tiny, None)

    def _estimate_marginal_density(self, t: pd.DataFrame) -> np.ndarray:
        """Approximate ``p(t)`` by averaging ``p(t | x)`` over sampled ``X`` rows.

        The method loops over treatment rows in batches to control memory use.
        For each batch, it forms the Cartesian product between the cached
        integration covariates and the batch of treatment values, evaluates the
        wrapped conditional estimator on that expanded dataset, reshapes the
        result back into ``(n_integration_rows, batch_size, ...)``, and then
        averages over the integration dimension.

        Parameters
        ----------
        t : pd.DataFrame
            Treatment rows whose marginal densities should be approximated.

        Returns
        -------
        np.ndarray
            Column vector of marginal-density estimates for the supplied
            treatment rows.
        """
        marginal_density_batches = []
        batch_size = max(1, min(len(t), self.max_batch_size))

        for start in range(0, len(t), batch_size):
            t_batch = t.iloc[start : start + batch_size].reset_index(drop=True)
            repeated_X, repeated_t = self._make_integration_batch(t_batch)
            conditional_density = self.conditional_density_estimator_.predict_density(
                repeated_X, repeated_t
            )

            marginal_density = conditional_density.reshape(
                self.n_integration_rows_,
                len(t_batch),
                -1,
            ).mean(axis=0)
            marginal_density_batches.append(marginal_density)

        return np.vstack(marginal_density_batches)

    def _make_integration_batch(
        self,
        t_batch: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build the expanded design used for one marginalization batch.

        This helper creates the Cartesian product between the cached integration
        rows of ``X`` and one batch of treatment rows. The resulting pair of
        data frames can be passed directly to the wrapped conditional estimator,
        which will then evaluate every treatment row against every cached
        covariate row.

        Parameters
        ----------
        t_batch : pd.DataFrame
            Consecutive slice of treatment rows being marginalized together.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Expanded covariate and treatment frames with matching row counts.
        """
        integration_index = np.repeat(
            np.arange(self.n_integration_rows_),
            len(t_batch),
        )
        treatment_index = np.tile(np.arange(len(t_batch)), self.n_integration_rows_)

        repeated_X = self.integration_X_.iloc[integration_index].reset_index(drop=True)
        repeated_t = t_batch.iloc[treatment_index].reset_index(drop=True)
        return repeated_X, repeated_t

    def _select_integration_rows(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select the covariate rows used for empirical marginalization.

        If ``n_samples`` is at least the training sample size, the full
        covariate matrix is reused. Otherwise, a reproducible simple random
        sample without replacement is taken so the marginal-density estimate is
        cheaper to evaluate while still approximating the empirical covariate
        distribution.

        Parameters
        ----------
        X : pd.DataFrame
            Training covariate matrix.

        Returns
        -------
        pd.DataFrame
            Cached covariate rows used inside the empirical integration step.
        """
        if self.n_samples >= len(X):
            return X.reset_index(drop=True)

        return X.sample(n=self.n_samples, replace=False, random_state=0).reset_index(
            drop=True
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return estimator configurations used by the object test suite.

        The default parameterization wraps the naive conditional density model
        and uses a reduced number of integration rows so generic object tests
        run quickly while still exercising the empirical-marginalization path.

        Parameters
        ----------
        parameter_set : str, default="default"
            Selector required by the skbase test interface. Only the default set
            is currently implemented.

        Returns
        -------
        list[dict]
            List of constructor argument dictionaries for test instances.
        """
        from skcausal.density.naive import NaiveDensityEstimator

        return [
            {
                "conditional_density_estimator": NaiveDensityEstimator(),
                "n_samples": 128,
            }
        ]
