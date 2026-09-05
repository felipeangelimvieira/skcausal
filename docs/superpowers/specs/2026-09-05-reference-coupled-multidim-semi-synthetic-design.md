# Reference-Coupled Multidimensional Semi-Synthetic Dataset Design

**Status:** Approved for implementation planning

## Goal

Replace the current multidimensional semi-synthetic assignment model with a
fixed observational dataset assembled from several real regression datasets.
Each source outcome becomes one continuous treatment dimension, whole source
rows are coupled through a Gaussian copula to introduce dependence, and fitted
mean regressors provide the shared predictors that induce confounding in the
synthetic outcome.

## Non-goals

- Do not sample treatments after construction.
- Do not expose a treatment-density oracle.
- Do not support categorical source outcomes in this version. A future source
  dataset may discretize a continuous outcome and define its ordering itself.
- Do not add treatment-effect interactions, heterogeneous effects, a general
  correlation matrix, or a new source-configuration abstraction.
- Do not change the scalar continuous or categorical semi-synthetic classes.

## Public API

```python
MultidimSemiSyntheticDataset(
    real_datasets,
    regressors,
    treatment_correlation=0.0,
    confounding_concentration=None,
    n_spline_knots=6,
    spline_degree=3,
    treatment_effect_scale=1.0,
    outcome_noise_scale=1.0,
    random_state=42,
)
```

`real_datasets` is a non-empty sequence of `BaseDataset` objects. Every source
must return non-empty covariates and exactly one finite numeric outcome column
with at least two distinct values. Its source treatment is ignored.

`regressors` is either one cloneable scikit-learn-style regressor, broadcast to
all sources, or a sequence with one regressor per source. A fitted regressor
must return one finite, nonconstant prediction per source row.

`treatment_correlation` is the scalar correlation of an exchangeable Gaussian
copula. For `P > 1` sources it must satisfy
`-1 / (P - 1) < treatment_correlation < 1`; for one source it must be zero.

`confounding_concentration=None` gives exact equal confounding weights. A
positive finite scalar draws symmetric Dirichlet weights once from the
constructor seed.

The existing synthetic-assignment parameters are removed from this class:
`confounding_loadings`, `n_levels`, `confounding_strength`,
`target_confounding_bias`, `treatment_only_strength`, `randomized_weight`,
`treatment_noise_scale`, `outcome_effect_scale`,
`effect_heterogeneity_scale`, and `treatment_interaction_scale`.

## Source preparation

Let source `j` contain `(X_j, Y_j)` with `n_j` rows and let

$$
n = \min_j n_j.
$$

The generator selects `n` rows without replacement from every larger source;
a source already of size `n` keeps all rows. Selection is controlled by a
dedicated stream derived from `random_state`.

For each selected source, standardize the outcome using its empirical mean and
population standard deviation,

$$
\widetilde Y_{ij} = \frac{Y_{ij} - \bar Y_j}{s_{Y_j}},
$$

clone and fit its regressor on `(X_j, \widetilde Y_j)`, and record

$$
\widehat m_j(x_j) = \widehat{\mathbb E}[\widetilde Y_j \mid X_j=x_j].
$$

No cross-fitting, residual model, or probabilistic-regressor protocol is added.
The behavior deliberately matches the existing point-regression model-induced
dataset.

## Gaussian-copula row coupling

For `P` sources define the exchangeable correlation matrix

$$
R_\rho = (1 - \rho)I_P + \rho\mathbf 1\mathbf 1^\top.
$$

Draw `n` independent rows

$$
G_i \sim \mathcal N(0, R_\rho).
$$

For each source `j`, sort the selected whole `(X_j, Y_j)` rows by `Y_j` and
assign them to combined rows in the rank order of `G_{ij}`. Ties in `Y_j`
are broken by the seeded coupling stream. No value inside a source row is
changed.

The released covariates concatenate the coupled source rows horizontally.
Source `j` column `name` becomes `d{j}_name`. The released treatment is

$$
T_i = (Y_{i1}, \ldots, Y_{iP}),
$$

with columns `t_0`, ..., `t_{P-1}` in their original source units. Thus every
treatment marginal and every within-source `(X_j, Y_j)` relationship is
preserved exactly. `treatment_correlation` is a latent Gaussian-copula
parameter, not a promise about finite-sample Pearson correlation. The fitted
object records the requested matrix plus achieved Pearson and Spearman
correlation matrices.

## Confounding scores and weights

Move each source prediction through the same row coupling as its source row,
then standardize it over the coupled sample:

$$
s_{ij} =
\frac{\widehat m_j(X_{ij}) - \overline{\widehat m_j}}
     {\operatorname{sd}(\widehat m_j)}.
$$

For the default configuration,

$$
w_j = \frac{1}{P}.
$$

For a configured positive concentration `alpha`, draw once

$$
w \sim \operatorname{Dirichlet}(\alpha\mathbf 1_P).
$$

The square roots of these weights are the score loadings. Form and standardize
the confounded baseline:

$$
b_i = \sum_{j=1}^P \sqrt{w_j}\,s_{ij},
\qquad
\mu_0(X_i) = \frac{b_i - \bar b}{\operatorname{sd}(b)}.
$$

When scores are uncorrelated, `w_j` is the fraction of pre-standardization
baseline variance contributed by source `j`. With correlated scores the
weights remain interpretable loadings, but the shared covariance prevents a
unique variance attribution. The fitted object exposes `confounding_weights_`,
the coupled standardized score matrix, and the baseline centering constants.

## Treatment response and observed outcome

Fit one `SplineTransformer` to each observed treatment column using the same
rules as `ModelInducedConfoundingRegressor`: uniformly placed knots, the
configured degree, no bias column, and continued extrapolation. If component
`j` has `K_j` basis columns, draw frozen coefficients

$$
\beta_{jk} \sim \mathcal N\!\left(0, \frac{1}{P K_j}\right).
$$

The `1 / P` factor keeps the scale of the additive response stable as the
number of treatment dimensions changes. Center the total response over the
observed treatments:

$$
g(t) = \lambda\left[
\sum_{j=1}^P B_j(t_j)^\top\beta_j
- \frac{1}{n}\sum_{i=1}^n\sum_{j=1}^P
  B_j(T_{ij})^\top\beta_j
\right],
$$

where `lambda` is `treatment_effect_scale`. The conditional mean and observed
outcome are

$$
m(t, x) = \mu_0(x) + g(t),
\qquad
Y_i = m(T_i, X_i) + \sigma_Y\varepsilon_i,
\quad \varepsilon_i \stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1).
$$

`predict_y(X, treatment)` evaluates `m(t, x)`. The inherited `predict` method
continues to average this known response surface over supplied covariates.
`get_grid(n)` keeps the existing approximate-size Cartesian behavior. With
`P` continuous components, each observed treatment range contributes
`max(2, round(n ** (1 / P)))` evenly spaced points and the method returns their
Cartesian product, yielding approximately `n` intervention rows.

## Fixed lifecycle

Construction materializes exactly one dataset. `random_state` determines row
selection, copula coupling, tie breaking, a possible Dirichlet draw, spline
coefficients, and outcome noise. Repeated `load()` calls return identical
`X`, `T`, and `Y`.

`sample(random_state)` raises `NotImplementedError` because treatment
replication contradicts the fixed-reference design. `log_prob(X, treatment)`
also raises `NotImplementedError` because ordinary mean regressors do not
define a conditional treatment density.

## Implementation boundaries

Reuse the existing multi-source loading hook in `BaseSemiSyntheticDataset`,
scikit-learn estimator cloning, `SplineTransformer`, and the repository's
backend coercion. Remove the multidimensional Gaussian-mixture density,
categorical-threshold, bias-calibration, and sampled-treatment machinery
instead of retaining a second mode.

Expected code changes are limited to:

- `src/skcausal/datasets/semi_synthetic.py` for the shared multi-source loading
  lifecycle needed by the multidimensional subclass;
- `src/skcausal/datasets/semi_synthetic_multidim.py` for coupling, fitted
  scores, fixed treatments, and the additive spline response;
- the existing multidimensional dataset tests and their small source fixtures;
- `docs/research/semi_synthetic_datasets.qmd` for the public mathematical
  description.

## Verification

Tests must demonstrate:

1. Each output treatment column is a permutation of its selected source target.
2. Every source covariate row remains paired with its own target after coupling.
3. Equal constructor seeds reproduce all fitted state and output frames; a
   changed seed can change selection, coupling, weights, effects, or noise.
4. Strong positive and negative requested copula correlations move achieved
   rank association in the corresponding direction.
5. Default weights are exactly `1 / P`; a finite concentration gives a seeded
   positive simplex draw; the baseline implements the documented square-root
   weighting and has zero mean and unit variance.
6. `predict_y` equals the documented baseline plus centered additive spline
   response, and zero outcome noise makes it equal the loaded outcome.
7. `sample` and `log_prob` are disabled.
8. Invalid sources, regressors, predictions, correlations, concentrations,
   spline parameters, and outcome-noise scales raise clear errors.
9. Existing scalar semi-synthetic datasets and the full dataset contract suite
   remain green.
