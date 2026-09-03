# Multi-Dimensional Semi-Synthetic DGP Design

## Status

Drafted on 2026-09-02 from a brainstorming session. Extends the design in
`2026-08-30-semi-synthetic-dgp-design.md` (as revised on 2026-09-02) with one
new concrete DGP. It changes the shared base only where noted and leaves the
categorical and scalar-continuous reference DGPs behaviorally unchanged.

## Goal

Add `MultidimSemiSyntheticDataset`, a semi-synthetic DGP whose treatment is a
vector of `n_treatments` components built from the real dataset, with:

- a frozen prognostic score fitted on the real outcome `y` as the confounding
  backbone, so every confounder is a function of real signal;
- per-component confounding loadings, so components can be strongly, mildly,
  or not confounded within one dataset;
- an exchangeable noise correlation between components, so correlation that is
  unexplained by covariates can be dialed independently of shared confounding;
- optional feature mimicking, where a component additionally tracks a named
  numeric covariate that is removed from the released `X`;
- optional discretization of any component into `K` ordered levels with an
  exact ordered-probit law; and
- the same exact oracles as the existing DGPs: `log_prob`, `predict_y`,
  `sample`, `get_grid`, and confounding-strength calibration.

## Non-goals

- Positive or bounded continuous domains. Correlated noise makes multivariate
  truncation expensive, so every continuous component lives on the real line.
- Correlated noise with two or more discretized components. The first version
  raises for this combination; a one-factor quadrature extension is described
  under "Extension boundaries".
- A user-supplied correlation matrix, non-uniform category marginals, or a
  user-supplied prognostic model. Each is a strict extension of a parameter.
- Any change to `ModelInducedConfoundingRegressor` or the two reference DGPs
  beyond the shared helper changes listed below.

## Public API

```python
MultidimSemiSyntheticDataset(
    real_dataset,
    n_treatments=2,
    confounding_loadings=None,        # scalar or length-k sequence; None -> all 1.0
    n_levels=None,                    # None, int >= 2, or length-k sequence of those
    target_columns=None,              # None, str, or length-k sequence of None/str
    treatment_correlation=0.0,        # exchangeable noise correlation in [0, 1)
    confounding_strength=1.0,
    target_confounding_bias=None,
    treatment_only_strength=0.0,
    randomized_weight=0.1,
    treatment_noise_scale=1.0,
    outcome_effect_scale=1.0,
    effect_heterogeneity_scale=0.5,
    treatment_interaction_scale=0.5,  # new: cross-component effect coefficients
    outcome_noise_scale=1.0,
    random_state=42,
)
```

Per-component parameters accept a scalar, which is broadcast to every
component, or a sequence of length `n_treatments`. `n_levels=None` for a
component means it is continuous. `target_columns=None` for a component means
it mimics no feature. Component `j` is released as column `t_j`, with
`column_types` set at preparation to `"continuous"` or `"categorical"` per
component. The vector treatment therefore reuses the `t_0`, `t_1`, ...
convention of `Synthetic2MultidimDataset`, and `target_columns_` records
which feature, if any, each component mimics.

Usage:

```python
dataset = MultidimSemiSyntheticDataset(
    real_dataset=real_dataset,
    n_treatments=3,
    confounding_loadings=[1.0, 0.5, 0.0],
    n_levels=[None, None, 3],
    target_columns=[None, "income", None],
    treatment_correlation=0.4,
    target_confounding_bias=0.5,
    random_state=42,
)
X, treatment, outcome = dataset.load()   # X excludes "income"
grid = dataset.get_grid(100)             # lattice on t_0, t_1 crossed with t_2 levels
```

`load`, `sample`, `predict_y`, `predict_curve`, `log_prob`, and `get_grid`
keep the semantics of the base design. `log_prob` returns the joint
conditional log-density of the released vector: a density in the continuous
components and a mass in the categorical ones.

## Source Split

The base `_prepare` gains one hook:

```python
def _split_source(self, X, treatment, outcome):
    """Return the covariates to release; keep any source columns needed."""
    return X
```

The base calls it after loading and validating the source frames and before
deriving `n`, storing `_covariates`, and calling `_prepare_dgp`. The default
is the identity, so the reference DGPs are unaffected. The new class:

- validates that the source outcome frame has exactly one numeric, non-boolean
  column with at least two distinct finite values, and keeps that column as
  the prognostic target;
- validates each named target column: it must exist in `X`, be numeric and
  non-boolean, appear at most once across components, and have at least two
  distinct finite values; and
- removes the target columns from the released `X`, requiring at least one
  covariate column to remain.

The released `X` retains the remaining source rows, values, names, and order.
All downstream schema checks, including `_check_and_transform_X` and the
NumPy-width shortcut, use the reduced frame.

## Frozen Scores

### Fitted confounder scores

`_fit_causal_score_map` gains a keyword `fitted_confounders=None`. When it is
an `(n, d)` array of standardized targets, the `confounders` role is no longer
a random sparse projection: each column becomes a ridge regression of the
target on the existing nonlinear library, with intercept absorbed by
centering the library and the target. The ridge penalty equals `n / 100` on
library columns standardized to unit variance, so shrinkage is scale-free;
zero-variance library columns receive coefficient zero. Rows whose target is
missing are excluded from the fit and predicted like any other row.

The fitted coefficients are stored as the `confounders` projection, so
`transform` on new rows, the standardization to zero mean and unit variance
over the source, and the residualization of the treatment-only scores against
`[q_C, q_P, q_M]` all work unchanged. The other roles keep their random
sparse definitions.

For the new class the fitted confounders are, in order, the standardized
source outcome `y` and the standardized mimicked features of the components
that name one. Their in-sample R² values are stored as `prognostic_fit_r2_`
and `feature_fit_r2_` for inspection.

### Baseline outcome surface

`_fit_baseline_surface` gains a keyword `confounder_coefficients=None`. When
given, those coefficients are used instead of random draws; the prognostic
coefficients, centering, and scaling are unchanged. The new class passes
`[1.0, g_1, ..., g_m]`, where `g_l ~ N(0, 1)` is frozen per mimicked feature.
The baseline is therefore

```
mu_0(x) = yhat(x) + sum_l g_l r_l(x) + f_P(q_P(x))      (standardized over the source)
```

with `yhat` the standardized prognostic score, `r_l` the standardized feature
scores, and `f_P` the random prognostic term. The random confounder block of
the reference DGPs is replaced by real signal; `q_P`, `q_Z`, and `q_M` keep
their roles.

## Assignment Law

Let `k` be `n_treatments`, `sigma` be `treatment_noise_scale`, `rho` be
`treatment_correlation`, `lambda` be the frozen confounding strength, `zeta`
be `treatment_only_strength`, and `omega` be `randomized_weight`.

The latent vector `z in R^k` has row-specific mean

```
eta_j(x) = lambda * ( c_j * yhat(x) + g_j * r_j(x) ) + zeta * b_j' q_Z(x)
```

where `c_j` is the user loading, `g_j r_j(x)` is present only for components
that mimic a feature (with the same `g_j` used in `mu_0`, so assignment and
outcome are sign-aligned), and `b_j` are frozen `N(0, 1/2)` coefficients on
the residualized treatment-only scores as in the categorical DGP.

The confounded latent law is `z | x ~ N(eta(x), sigma^2 R)` with the
exchangeable correlation `R = (1 - rho) I + rho 11'`, which is positive
definite for `rho in [0, 1)`. The randomized law is `N(0, sigma^2 R)`. Each
row uses the randomized law with probability `omega`, so the joint law is a
two-component mixture per row, as in the scalar DGP.

At `lambda = 0` and `zeta = 0` every row mean is zero, the law does not depend
on `x`, and oracle confounding is exactly zero. This preserves the invariant
of the revised base design.

### Released continuous components

A continuous component is released as `a_j = m_j + s_j * z_j`. For a
component mimicking feature `f`, `m_j` and `s_j` are the source mean and
standard deviation of `f`, so the released marginal sits on the feature's
scale. Otherwise `m_j = 0` and `s_j = 1`. The inverse transform is affine and
its log-Jacobian is the constant `-log s_j`.

### Released categorical components

A component with `K` levels is released as the string label `str(k)` of the
interval `tau_{k-1} < z_j <= tau_k`, with `tau_0 = -inf`, `tau_K = +inf`, and
interior cut points at the `1/K, ..., (K-1)/K` quantiles of the component's
marginal latent law under the frozen assignment. The marginal CDF is the
row-average of the mixture normal CDF; quantiles are found by bracketed
root-finding, reusing the one-dimensional marginal helpers lifted from the
scalar DGP to module level. Cut points depend on strength and are recomputed
for each candidate during calibration, then frozen.

Conditional on `x`, the level mass is the ordered-probit difference
`Phi((tau_k - eta_j) / sigma) - Phi((tau_{k-1} - eta_j) / sigma)` under
either mixture component.

### Joint log-probability

Split components into continuous indices `C` and categorical indices `D`. For
each mixture component with mean `mu` (either `eta(x)` or `0`) and covariance
`Sigma = sigma^2 R`:

- the continuous part contributes the multivariate normal log-density of
  `z_C` under `N(mu_C, Sigma_CC)`, using the closed-form Cholesky factor of the
  exchangeable matrix;
- conditional on `z_C`, `z_D` is Gaussian with mean
  `mu_D + Sigma_DC Sigma_CC^{-1} (z_C - mu_C)` and covariance
  `Sigma_DD - Sigma_DC Sigma_CC^{-1} Sigma_CD`;
- with `|D| <= 1`, or with `rho = 0` and any `|D|`, the categorical mass is a
  product of one-dimensional `Phi` differences, computed with `log_ndtr` on the
  side of the interval with small mass; and
- `rho > 0` with `|D| >= 2` is rejected at construction in this version.

The two mixture components are combined with the stable log-mixture helper,
and the continuous Jacobian constant is added. Unknown category labels give
`-inf`; malformed treatment inputs raise a validation error. Non-finite
continuous values give `-inf`.

### Sampling

Rows draw the mixture indicator, then `z = mu + sigma * L eps` with `L` the
Cholesky factor of `R` and `eps` standard normal, then apply the released
transform per component. Sampling and `log_prob` use the same law exactly.

## Outcome Surface

The outcome keeps the unified Gaussian form of the base design, with a
component-wise basis plus cross-component terms:

```
m(a, x) = mu_0(x)
        + sum_j B_j(a_j)' beta_j
        + sum_{j < l} phi_j(a_j)' Theta_jl phi_l(a_l)
        + sum_j B_j(a_j)' Gamma_j q_M(x)
```

- For a continuous component, `B_j` is the scalar DGP's bounded-growth basis
  in standardized latent units `v_j = z_j / sigma`: linear, saturating
  curvature, sinusoid, and two tertile hinges. The basis is standardized over
  the central region from the 1% and 99% quantiles of the randomized latent
  law and reference-subtracted at `z = 0`. Because every continuous component
  shares `sigma`, the standardization is computed once and shared.
- For a categorical component, `B_j` is the `K - 1` indicator basis against
  level `"0"`, as in the categorical DGP.
- `phi_j` is `tanh(v_j)` for a continuous component and the indicator vector
  for a categorical one, so every cross term is bounded and vanishes at the
  reference treatment. `Theta_jl` has one coefficient per pair of basis
  entries, drawn `N(0, treatment_interaction_scale^2)`. With `k = 1` there are
  no cross terms.
- `beta_j ~ N(0, outcome_effect_scale^2)` and
  `Gamma_j ~ N(0, effect_heterogeneity_scale^2)` per basis entry.

The outcome depends only on the released values. A categorical component's
underlying latent never enters `m`, so the released treatment is not a
coarsening of a hidden cause and ignorability holds given `X`.

`_outcome_mean` in the base is reused by stacking the component bases into
one basis matrix with concatenated main effects and modifier coefficients; the
cross-term contribution is added by the subclass.

Observed outcomes add `outcome_noise_scale` Gaussian noise, as before.

## Confounding Calibration

The scalar summary is the same root-mean-square naive-curve distortion under
the marginal treatment law, normalized by `sd(mu_0)`:

```
bias^2 = E_{a ~ f_A} [ (mu_obs(a) - mu_do(a))^2 ]
```

Quadrature over a lattice does not scale past two dimensions, so the metric
is a Monte Carlo estimate under the marginal with frozen common random
numbers and an exactly enumerated discrete part:

1. At preparation, draw once from the structural RNG a fixed design of
   `M = 1024` triples: a source row index `i_m`, a latent noise vector
   `eps_m ~ N(0, R)`, and a mixture uniform `u_m`.
2. For a candidate strength, the continuous coordinates of design point `m`
   are `z_{C,m} = mu_{C,m} + sigma * eps_{C,m}` with
   `mu_m = 0` if `u_m < omega` else `eta_{i_m}`. These are draws from the
   marginal of the continuous components, and they move continuously with
   strength.
3. For every design point and every combination `kappa` of categorical
   levels, evaluate `g(z_{C,m}, kappa | X_i)` for all source rows in one
   vectorized, chunked pass. Their row-average is the marginal
   `f_A(z_{C,m}, kappa)`, and their normalization gives the observational
   weights for `mu_obs`.
4. `mu_obs - mu_do` is computed from weighted `mu_0` and weighted modifier
   scores, as in the scalar DGP. Main effects and cross terms do not depend
   on `x`, so they cancel in the distortion; only the baseline and the
   modifier interaction evaluated at the basis of `(z_{C,m}, kappa)` remain.
5. The estimate is
   `sum_m sum_kappa P(kappa | z_{C,m}) Delta(z_{C,m}, kappa)^2 / M`, with
   `P(kappa | z_{C,m}) = f_A(z_{C,m}, kappa) / sum_kappa' f_A(z_{C,m}, kappa')`.

Enumerating the discrete part keeps the metric continuous in strength, so the
existing ladder-plus-Brent search in `_resolve_confounding_strength` applies
unchanged. Cut points are recomputed at each candidate strength before the
evaluation. When there are no continuous components the design reduces to the
enumeration alone and is exact.

At `randomized_weight = 1`, or at zero strength with zero treatment-only
strength, the metric is exactly zero and no evaluation is performed. The
stored `confounding_strength_`, `confounding_bias_`, and
`confounding_bias_ratio_` have the same meaning as in the reference DGPs, and
the achieved bias is reported with the same estimator used for calibration.
The Monte Carlo standard error is of order `1 / sqrt(M)` relative and is
documented as the calibration resolution.

## Grid

`get_grid(n)` returns the cross join of:

- for each continuous component, `max(2, round(n ** (1 / k_C)))` equally
  spaced released values spanning the union of the shared central region and
  the 1% to 99% quantiles of that component's latent marginal under the
  frozen assignment, where `k_C` is the number of continuous components; and
- for each categorical component, all of its levels.

With no continuous components the grid is the level lattice and `n` is
ignored. Grid rows are valid intervention rows for `predict_curve`.

## Validation

- `n_treatments` is an integer of at least one.
- `confounding_loadings` entries are finite; `n_levels` entries are `None` or
  integers of at least two; `target_columns` entries are `None` or strings;
  sequences have length `n_treatments`.
- `treatment_correlation` is finite and in `[0, 1)`; with two or more
  categorical components it must be zero.
- `treatment_noise_scale` is strictly positive; `treatment_interaction_scale`
  is nonnegative; the shared parameters go through
  `_validate_shared_dgp_parameters`.
- The source outcome and target columns satisfy the "Source Split" rules.
- Oracle calls accept NumPy, pandas, and Polars inputs. Treatment frames must
  contain exactly the released treatment columns; categorical columns are
  compared as strings and continuous columns as floats.

## Files and Exports

```text
src/skcausal/datasets/
    semi_synthetic.py             # _split_source hook; fitted_confounders;
                                  # confounder_coefficients; lifted 1-D marginal helpers
    semi_synthetic_continuous.py  # uses the lifted helpers; behavior unchanged
    semi_synthetic_multidim.py    # MultidimSemiSyntheticDataset
    __init__.py                   # export
tests/datasets/
    test_semi_synthetic_multidim_dgp.py
    test_semi_synthetic_helpers.py   # fitted-confounder and surface-coefficient cases
docs/research/
    semi_synthetic_datasets.qmd      # new step: multi-dimensional and mixed treatments
    dataset_catalog.qmd              # catalog entry
```

## Testing

1. Released `X` drops the target columns and preserves the remaining columns,
   rows, and order; the source dataset is not mutated.
2. Validation errors for bad `n_treatments`, per-component sequence lengths,
   unknown or non-numeric target columns, duplicate targets, no remaining
   covariates, a missing or non-numeric source outcome, correlation out of
   range, and correlation with two categorical components.
3. Fitted confounder scores are reproducible from the seed, transform new rows
   identically to source rows, and report R² in `[0, 1]`.
4. Continuous-only joint density integrates to one on a two-dimensional
   lattice for several rows, at `rho = 0` and `rho = 0.6`.
5. One categorical plus one continuous component: masses sum to one across
   levels at fixed continuous value, and `log_prob` matches brute-force
   numerical integration of the latent law over each cut interval.
6. Sampled component correlation tracks `rho` at zero strength, and a zero
   loading with zero treatment-only strength yields a component independent of
   `X`.
7. Zero oracle bias at `randomized_weight = 1` and at zero strength; the
   metric increases with strength on a small sweep; calibration reaches a
   target within tolerance for a two-component and for a mixed configuration.
8. `predict_y` equals the conditional mean implied by the sampler, depends
   only on released values, and honors the reference treatment.
9. `get_grid` has the expected shape and dtypes, including the categorical-only
   case; `column_types` reflects the configuration.
10. `sample` reproducibility and non-mutation of `load` outputs.
11. Discovery, cloning, parameter inspection, and `get_test_params`, with one
    continuous-only and one mixed configuration on `KangSchaferContinuous`.
12. The existing helper tests extended for `fitted_confounders` and
    `confounder_coefficients`; the full test suite passes.

## Extension Boundaries

- **Correlated noise with several categorical components.** The exchangeable
  correlation is a one-factor model, so the conditional categorical mass is a
  one-dimensional integral over the shared factor of a product of `Phi`
  differences. Gauss-Hermite quadrature with a fixed node count would make it
  vectorizable; this is the planned path when the restriction is lifted.
- **General correlation matrices** replace `R` in the Cholesky and
  conditioning steps without touching the rest.
- **Non-uniform category marginals** replace the equal-mass quantiles with a
  target cumulative distribution.
- **A user-supplied prognostic model** replaces the ridge fit inside
  `_split_source` and `_prepare_dgp`; the frozen-function contract is the same.
- **Positive or bounded components** require a truncation scheme that is
  compatible with correlated noise and are out of scope.
