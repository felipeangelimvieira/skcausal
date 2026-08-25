# CBPS density estimators — design

Date: 2026-08-25. Branch: `feat/cbps`.

## Goal

Add pure-Python (numpy/scipy) implementations of the Covariate Balancing
Propensity Score (CBPS) family to `skcausal.density`, exposed through the
existing `BaseDensityEstimator` interface, and verified against the R
`CBPS` package (kosukeimai/CBPS).

## Scope

In scope — the three parametric routines behind R's `CBPS()`:

| R routine | Python | Treatment |
| --- | --- | --- |
| `CBPS.2Treat` (ATE, ATT=1/2) | `CBPSCategorical` with 2 levels | binary |
| `CBPS.3Treat`, `CBPS.4Treat` | `CBPSCategorical` with J ≥ 3 levels | multi-valued |
| `CBPS.Continuous` (CBGPS) | `CBPSContinuous` | continuous |

Out of scope: `npCBPS`, `hdCBPS`, `CBMSM`, `CBIV`, `CBPSOptimal`, survey
`sample.weights` (the skcausal `fit(X, t)` contract has no weight argument),
and asymptotic variance (`var`) output.

Assumptions (made autonomously, easy to revisit):

- Covariates `X` must be numeric. An intercept is added internally, as R's
  formula interface does.
- `CBPSCategorical` generalises to any number of levels J ≥ 2. R's `over`
  objective uses the model-implied covariance of the moment conditions, so it
  is invariant to the contrast basis; we use "level j vs baseline" contrasts
  `T_j/π_j − T_0/π_0`, which coincide with R's binary weights and produce the
  same optimum as R's hand-coded 3- and 4-level contrasts.
- Optimizer: `scipy.optimize.minimize(method="BFGS")` (R uses `optim` BFGS).
  Optima match R to optimizer tolerance, not bit-for-bit.

## Public API

```python
from skcausal.density.cbps import CBPSCategorical, CBPSContinuous

CBPSCategorical(method="over", att=0, twostep=True, max_iter=1000,
                standardize=True)
CBPSContinuous(method="over", twostep=True, max_iter=1000, standardize=True,
               density_kind="conditional")
```

- `method`: `"over"` (score + balance moments, GMM) or `"exact"` (balance
  moments only), as in R.
- `att`: 0 = ATE; 1 = ATT treating the second level as treated; 2 = ATT
  treating the first level as treated. Binary only (R behaviour).
- `twostep`: fixed weighting matrix from the MLE initial values (analytic
  gradients) vs. continuous-updating GMM (numerical gradients).
- `standardize`: normalises `fitted_weights_` as R does; it does not affect
  `predict_density`.
- `density_kind` (continuous only): `"conditional"` returns the Gaussian
  `p(t | x)` on the original treatment scale; `"stabilized"` returns
  `p(t | x) / p(t)` with `p(t)` the Gaussian fitted to the marginal of `t`
  (the inverse of R's `weights` before standardisation).

`predict_density(X, t)` returns `p(t_i | x_i)` (categorical: the fitted
probability of the observed level; continuous: the Gaussian density).

Fitted attributes mirroring R output: `coefficients_` (original-scale, with
intercept row), `fitted_values_` (probabilities per level, or Gaussian
density on the standardised scale), `fitted_weights_`, `J_`, `mle_J_`,
`converged_`, `classes_` (categorical), `sigmasq_` (continuous).

## Architecture

```
src/skcausal/density/cbps/
  __init__.py        exports CBPSCategorical, CBPSContinuous
  _design.py         DesignStandardizer: intercept + z-score + SVD, forward
                     transform for new X, coefficient back-transform
  _glm.py            logistic / multinomial / linear MLE initialisers
  _optim.py          shared helpers: scale search on [0.8, 1.1], BFGS wrapper,
                     "pick best of MLE-init vs balance-init" logic
  _multinomial.py    fit_cbps_multinomial(U, T_onehot, ...) -> CBPSResult
                     (J >= 2, ATE; binary ATE is J = 2)
  _binary_att.py     fit_cbps_binary_att(U, treat, ...)     -> CBPSResult
  _continuous.py     fit_cbps_continuous(U, treat, ...)     -> CBPSResult
  categorical.py     CBPSCategorical(BaseDensityEstimator)
  continuous.py      CBPSContinuous(BaseDensityEstimator)
```

Core fitters are plain functions on numpy arrays (already in the SVD-U
space) returning a small `CBPSResult` dataclass. Estimator classes handle
polars → numpy conversion, level encoding, standardisation, and prediction.

## Algorithms (mirroring R)

Preprocessing (`CBPS.fit`): X ← [1, X]; z-score non-intercept columns with
ddof=1; `U, d, V = svd(X)`; run the fitter on `U`; map
`beta_orig = V diag(1/d) beta_U`, divide slopes by sd, adjust intercept.

Multinomial (ATE): probabilities via softmax with baseline level 0, clipped
at 1e-6 then renormalised. Moments per unit: scores `T_j − π_j` (j ≥ 1) and
balance `T_j/π_j − T_0/π_0` (j ≥ 1), each multiplied by `u_i`. Weighting
matrix V from model-implied covariances:
`Cov(score_a, score_b) = π_a δ_ab − π_a π_b`, `Cov(score_a, bal_j) = δ_aj`,
`Cov(bal_j, bal_l) = δ_jl/π_j + 1/π_0`. Init from multinomial MLE, rescale
by the scalar minimising the GMM loss on [0.8, 1.1], optimise balance loss
(`||g_bal||²`) and, for `over`, the GMM loss from both starts, keep the
smaller. Fall back to MLE if both J and balance loss are worse than MLE.
Weights `Σ_j T_j/π_j`, standardised to sum to 1 per level.

Binary ATT: as `CBPS.2Treat` with `w = n/n_t (T − π)/(1 − π)` and the R
weighting matrix.

Continuous: whiten non-constant columns of U with the Cholesky factor of
their covariance and centre; standardise t; Gaussian log-density clipped to
[log 1e-6, log(1 − 1e-6)]; balance moment `E[u · t̃ · f(t̃)/f(t̃|u)]` with
`f(t̃)` the standard normal; variance moment `E[(t̃ − uβ)²/σ² − 1]`. Same
init/scale/optimise/pick logic; back-transform `beta` and `σ²` to the
original scale.

## Testing

- Unit tests (TDD, written first): design transform round-trips; MLE
  initialisers against sklearn/closed forms; estimator tags and shapes;
  probabilities sum to 1; exact-method balance property (weighted covariate
  means equal across levels / weighted covariance of X and t ≈ 0);
  stabilized = conditional / marginal Gaussian; parameter validation.
- The generic `TestAllDensityEstimators` suite picks the classes up
  automatically (`get_test_params`).
- R compatibility: `tests/density/cbps/fixtures/generate_fixtures.R` fits
  the R package on CSV datasets generated by `make_fixture_data.py` and
  writes JSON (coefficients, fitted values, weights, J, mle.J). Python tests
  assert (a) our objective evaluated at R's coefficients equals R's `J`
  (validates the moment/weighting code), (b) our optimum's objective is
  ≤ R's `J` (+ tolerance) (validates the optimiser), and (c) fitted values
  and weights match within tolerance. Fixtures are committed so the tests
  run without R.
