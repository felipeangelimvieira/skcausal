from typing import ClassVar

import numpy as np
import polars as pl

from skcausal.datasets.semi_synthetic import (
    BaseSemiSyntheticDataset,
    _calibrate_softmax_intercepts,
    _softmax,
    _validate_shared_dgp_parameters,
)

__all__ = ["CategoricalSemiSyntheticDataset"]


class CategoricalSemiSyntheticDataset(BaseSemiSyntheticDataset):
    r"""Semi-synthetic DGP with one categorical treatment.

    The confounded assignment law is a softmax over
    ``alpha_k + confounding_strength * l_k s(x) + treatment_only_strength *
    B_k z(x)``, where ``s`` is the standardized confounder contribution to the
    baseline outcome, ``l_k`` are centered equally spaced loadings, and ``z``
    is a treatment-only score residualized against the outcome scores. The
    intercepts ``alpha`` are calibrated so the empirical marginal matches
    ``target_probabilities``; the released law mixes this with the target
    marginal using ``randomized_weight``.

    The oracle confounding bias is the largest pairwise distortion between
    naive category means ``E[Y | A = k]`` and causal means ``E[Y(k)]``, in
    units of the baseline-outcome standard deviation.

    Parameters
    ----------
    real_dataset : BaseDataset
        Dataset supplying the covariate rows retained by this DGP.
    n_treatments : int, default=2
        Number of treatment levels, released as labels ``"0"`` to ``"K-1"``.
    target_probabilities : array-like or None, default=None
        Target empirical marginal of the assignment; uniform when omitted.
    confounding_strength : float, default=1.0
        Nonnegative scale of the outcome-predictive assignment signal.
    target_confounding_bias : float or None, default=None
        Optional nonnegative normalized oracle-bias calibration target. The
        smallest strength reaching the target is frozen.
    treatment_only_strength : float, default=0.0
        Nonnegative scale of the residualized treatment-only predictors. These
        scores add assignment predictability that is orthogonal, to second
        order, to the outcome surface; because real covariates are dependent, a
        small residual confounding floor that grows roughly quadratically with
        this value remains at ``confounding_strength=0``.
    randomized_weight : float, default=0.1
        Weight in ``[0, 1]`` of the covariate-independent mixture component.
    outcome_effect_scale : float, default=1.0
        Nonnegative scale of treatment main-effect coefficients.
    effect_heterogeneity_scale : float, default=0.5
        Nonnegative scale of treatment-by-covariate coefficients.
    outcome_noise_scale : float, default=1.0
        Nonnegative Gaussian outcome-noise standard deviation.
    random_state : int, default=42
        Seed controlling frozen structure and the initial sample.
    """

    column_types: ClassVar = {"t": "categorical"}

    def __init__(
        self,
        real_dataset,
        n_treatments=2,
        target_probabilities=None,
        confounding_strength=1.0,
        target_confounding_bias=None,
        treatment_only_strength=0.0,
        randomized_weight=0.1,
        outcome_effect_scale=1.0,
        effect_heterogeneity_scale=0.5,
        outcome_noise_scale=1.0,
        random_state=42,
    ):
        if isinstance(n_treatments, bool) or not isinstance(
            n_treatments, (int, np.integer)
        ):
            raise TypeError("n_treatments must be an integer.")
        if n_treatments < 2:
            raise ValueError("n_treatments must be at least 2.")
        probabilities = (
            np.full(n_treatments, 1.0 / n_treatments)
            if target_probabilities is None
            else np.asarray(target_probabilities, dtype=float)
        )
        if (
            probabilities.shape != (n_treatments,)
            or np.any(probabilities <= 0.0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "target_probabilities must contain one positive value per treatment "
                "and sum to one."
            )
        _validate_shared_dgp_parameters(
            confounding_strength=confounding_strength,
            target_confounding_bias=target_confounding_bias,
            treatment_only_strength=treatment_only_strength,
            randomized_weight=randomized_weight,
            outcome_effect_scale=outcome_effect_scale,
            effect_heterogeneity_scale=effect_heterogeneity_scale,
            outcome_noise_scale=outcome_noise_scale,
        )
        self.n_treatments = n_treatments
        self.target_probabilities = target_probabilities
        self.confounding_strength = confounding_strength
        self.target_confounding_bias = target_confounding_bias
        self.treatment_only_strength = treatment_only_strength
        self.randomized_weight = randomized_weight
        self.outcome_effect_scale = outcome_effect_scale
        self.effect_heterogeneity_scale = effect_heterogeneity_scale
        self.outcome_noise_scale = outcome_noise_scale
        self._validated_target_probabilities = probabilities
        super().__init__(real_dataset=real_dataset, random_state=random_state)

    def _prepare_dgp(self, X, rng):
        scores = self._fit_causal_structure(X, rng)
        n_treatments = int(self.n_treatments)
        self.treatment_levels_ = [str(index) for index in range(n_treatments)]
        self.target_probabilities_ = self._validated_target_probabilities.copy()

        self.confounding_loadings_ = np.linspace(-1.0, 1.0, n_treatments)
        self.confounding_loadings_ -= self.confounding_loadings_.mean()
        self.treatment_only_coefficients_ = rng.normal(
            scale=1.0 / np.sqrt(2.0), size=(2, n_treatments)
        )
        self.treatment_only_coefficients_ -= self.treatment_only_coefficients_.mean(
            axis=1, keepdims=True
        )
        self.outcome_main_effects_ = rng.normal(
            scale=self.outcome_effect_scale, size=n_treatments - 1
        )
        self.outcome_modifier_coefficients_ = rng.normal(
            scale=self.effect_heterogeneity_scale, size=(n_treatments - 1, 2)
        )
        self._source_category_outcome_means = np.column_stack(
            [
                self.source_mu0_,
                self.source_mu0_[:, None]
                + self.outcome_main_effects_[None, :]
                + scores.modifiers @ self.outcome_modifier_coefficients_.T,
            ]
        )

        self._resolve_confounding_strength(
            self._bias_at_strength,
            confounding_strength=self.confounding_strength,
            target_confounding_bias=self.target_confounding_bias,
            randomized_weight=self.randomized_weight,
        )
        self.assignment_intercepts_ = self._calibrated_intercepts(
            self.confounding_strength_
        )

    def _confounded_logits(self, scores, strength):
        _, signal = self.baseline_surface_.evaluate(scores)
        treatment_only = scores.treatment_only @ self.treatment_only_coefficients_
        return (
            strength * signal[:, None] * self.confounding_loadings_[None, :]
            + self.treatment_only_strength * treatment_only
        )

    def _calibrated_intercepts(self, strength):
        logits = self._confounded_logits(self.source_scores_, strength)
        return _calibrate_softmax_intercepts(logits, self.target_probabilities_)

    def _mixed_probabilities(self, logits, intercepts):
        confounded = _softmax(logits + intercepts)
        return (
            1.0 - self.randomized_weight
        ) * confounded + self.randomized_weight * self.target_probabilities_

    def _bias_at_strength(self, strength):
        if self.randomized_weight == 1.0 or (
            strength == 0.0 and self.treatment_only_strength == 0.0
        ):
            return 0.0, 0.0
        logits = self._confounded_logits(self.source_scores_, strength)
        probabilities = self._mixed_probabilities(
            logits, self._calibrated_intercepts(strength)
        )
        outcome_means = self._source_category_outcome_means
        causal_means = outcome_means.mean(axis=0)
        observational_means = (probabilities * outcome_means).sum(
            axis=0
        ) / probabilities.sum(axis=0)
        bias = float(np.ptp(observational_means - causal_means))
        return bias, bias / self.baseline_standard_deviation_

    def _probabilities(self, X):
        logits = self._confounded_logits(self._scores(X), self.confounding_strength_)
        return self._mixed_probabilities(logits, self.assignment_intercepts_)

    def _sample_treatment(self, X, rng):
        probabilities = self._probabilities(X)
        uniforms = rng.random(probabilities.shape[0])
        cdf = np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        indices = (uniforms[:, None] > cdf).sum(axis=1)
        levels = np.asarray(self.treatment_levels_, dtype=str)[indices]
        return pl.DataFrame({"t": levels})

    def _treatment_indices(self, treatment, *, allow_unknown):
        labels = treatment.get_column("t").cast(pl.Utf8)
        indices = labels.replace_strict(
            {level: index for index, level in enumerate(self.treatment_levels_)},
            default=-1,
            return_dtype=pl.Int64,
        ).to_numpy()
        if not allow_unknown and np.any(indices < 0):
            unsupported = sorted(set(labels.to_numpy()[indices < 0].tolist()))
            raise ValueError(f"Unsupported treatment levels: {unsupported}.")
        return indices

    def _log_prob(self, X, treatment):
        probabilities = self._probabilities(X)
        indices = self._treatment_indices(treatment, allow_unknown=True)
        log_probabilities = np.full(indices.shape, -np.inf, dtype=float)
        valid = indices >= 0
        with np.errstate(divide="ignore"):
            log_probabilities[valid] = np.log(
                probabilities[np.arange(indices.size)[valid], indices[valid]]
            )
        return log_probabilities

    def _treatment_basis(self, treatment):
        indices = self._treatment_indices(treatment, allow_unknown=False)
        return (indices[:, None] == np.arange(1, self.n_treatments)[None, :]).astype(
            float
        )

    def _predict_y(self, X, treatment):
        return self._outcome_mean(
            self._scores(X),
            self._treatment_basis(treatment),
            self.outcome_main_effects_,
            self.outcome_modifier_coefficients_,
        )

    def _sample_outcome(self, mean, X, treatment, rng):
        return self._gaussian_outcome(mean, self.outcome_noise_scale, rng)

    def _get_grid(self, n):
        return self.get_levels()

    def get_levels(self):
        return self._coerce_treatment_frame(pl.DataFrame({"t": self.treatment_levels_}))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.datasets.kang_schafer import KangSchaferContinuous

        return [
            {
                "real_dataset": KangSchaferContinuous(n=64, random_state=3),
                "random_state": 7,
            }
        ]
