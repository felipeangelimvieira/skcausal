# Semi-Synthetic Observational DGP Design

## Status

Approved in conversation on 2026-08-30. This document defines the design to be
implemented; it does not describe the current package behavior.

## Goal

Add a small, extensible base for semi-synthetic observational datasets that:

- takes an existing `BaseDataset` object as its source of real covariates;
- preserves the source covariate rows and empirical distribution;
- simulates treatment and outcome from a frozen structural DGP;
- exposes the exact conditional assignment law \(g(a \mid x)\);
- exposes the exact conditional outcome mean \(m(a, x)\) through the existing
  `predict_y` API;
- supports categorical and scalar-continuous reference DGPs initially; and
- can be subclassed for vector, mixed, ordinal, count, alternative-outcome, and
  hidden-confounding DGPs without changing the base contract.

The design should fit the existing dataset philosophy: dataset objects own their
generation lifecycle, return `(X, treatment, outcome)` from `load`, use Polars at
the internal dataset boundary, participate in `skbase` cloning and discovery,
and expose average response curves through `predict_curve`/`predict`.

## Non-goals

The first implementation will not add separate public objects for score
generation, treatment assignment, outcome generation, calibration, or
benchmark orchestration. It will not refactor the existing
`SemiSyntheticRegressor` or `SemiSyntheticClassifier` classes.

Only categorical and scalar-continuous treatment implementations and Gaussian
continuous outcomes are built in initially. Other mechanisms are extensions of
the same protected base-class contract, not reasons to expand that contract.

## Architecture

### One base class for a complete DGP

`BaseSemiSyntheticDataset` extends `BaseSyntheticDataset`. A concrete subclass
represents one coherent DGP and owns both oracle functions, rather than
composing separate assignment and outcome objects. This keeps deliberately
coupled quantities, such as shared prognostic scores and targeted selection, in
one place.

The class is initialized with a real dataset object:

```python
class BaseSemiSyntheticDataset(BaseSyntheticDataset):
    def __init__(
        self,
        real_dataset: BaseDataset,
        random_state: int = 42,
    ):
        ...
```

During preparation it clones the supplied object and calls its existing
contract:

```python
self.real_dataset_ = self.real_dataset.clone()
X, _, _ = self.real_dataset_.load()
```

Only `X` is used. The source treatment and outcome are ignored by the built-in
DGPs. The released covariate frame retains the source rows, values, names, and
order. Encoding, imputation, standardization, and latent scores are stored only
as internal fitted state.

Cloning prevents the generator from mutating the caller's dataset object and
matches the package's nested-object conventions.

### Protected subclass contract

The base class owns lifecycle, validation, backend coercion, random-stream
separation, and public wrappers. Concrete subclasses implement the causal
mathematics through these hooks:

```python
class BaseSemiSyntheticDataset(BaseSyntheticDataset):
    def _prepare_dgp(self, X, rng):
        """Freeze encoders, scores, structural coefficients, and calibration."""

    def _sample_treatment(self, X, rng):
        """Draw a treatment frame from the frozen assignment law."""

    def _log_prob(self, X, treatment):
        """Evaluate exact conditional log-mass or log-density row by row."""

    def _predict_y(self, X, treatment):
        """Evaluate the frozen conditional outcome mean row by row."""

    def _sample_outcome(self, mean, X, treatment, rng):
        """Draw outcomes around the conditional mean."""

    def _get_grid(self, n):
        """Return valid intervention rows for this DGP."""
```

The exact private method names may follow existing `BaseSyntheticDataset`
conventions during implementation, but these responsibilities must remain
separate. In particular, structural preparation must not be repeated when a
new stochastic replication is drawn.

A mixin is intentionally not used. The source data, frozen DGP state, oracle
functions, and sampling lifecycle share state and ordering; a base class makes
those invariants explicit. Small numerical operations shared by concrete DGPs
remain private functions rather than public framework classes.

## Public API

The new base adds only two public capabilities to the existing synthetic
dataset API:

```python
dataset.log_prob(X, treatment)
dataset.sample(random_state)
```

Normal usage is:

```python
dataset = CategoricalSemiSyntheticDataset(
    real_dataset=real_dataset,
    confounding_strength=1.0,
    randomized_weight=0.2,
    random_state=42,
)

X, treatment, outcome = dataset.load()

mean = dataset.predict_y(X, treatment)
curve = dataset.predict_curve(X, dataset.get_grid())
log_probability = dataset.log_prob(X, treatment)

X_rep, treatment_rep, outcome_rep = dataset.sample(random_state=7)
```

The method semantics are:

- Construction freezes the structural DGP and generates the initial sample.
- Repeated `load()` calls return the same initial sample.
- `sample(random_state)` draws new treatment and outcome noise while retaining
  the same `X`, encoders, latent score definitions, coefficients, assignment
  law, and outcome surface.
- `sample` does not mutate the frames subsequently returned by `load`.
- Equal sampling seeds return equal replications.
- `predict_y(X, treatment)` is the exact \(m(a,x)\) and retains the existing
  `(n_samples, n_outcomes)` return convention.
- `log_prob(X, treatment)` evaluates paired rows and returns an
  `(n_samples, 1)` float array. It is a log-probability mass for categorical
  treatments and a log-density for continuous treatments.
- `get_grid(n)` returns valid intervention rows for the concrete DGP.
- `predict_curve` and its `predict` alias continue to compute empirical-population
  average responses using the source covariates.

`prepare(n=...)` is invalid because sample size comes from `real_dataset`.
Calling parameterless `prepare()` may reconstruct the same structural DGP and
initial sample deterministically from `random_state`, consistent with existing
semi-synthetic datasets.

## Randomness and Replications

Structural randomness and initial sampling randomness are derived from separate
child streams of the constructor's `random_state`. Structural randomness covers
feature selection, nonlinear score definitions, DGP coefficients, and any
unit-level latent variables that are part of the frozen condition. Sampling
randomness covers treatment draws and outcome noise.

`sample(random_state)` creates fresh treatment and outcome generators from the
given seed and never consumes or modifies the structural generator. This gives
replications a common structural DGP without introducing a benchmark runner or
replication artifact format.

## Internal Causal Roles

The built-in DGPs encode and standardize a private copy of `X`, then freeze
sparse nonlinear functions producing:

\[
q_C(X),\quad q_P(X),\quad q_Z(X),\quad q_M(X).
\]

The roles are structural:

| Internal score | Assignment | Baseline outcome | Treatment effect |
| --- | ---: | ---: | ---: |
| \(q_C\) | yes | yes | no |
| \(q_P\) | no | yes | no |
| \(q_Z\) | yes | no | no |
| \(q_M\) | no | no | yes |

The internal helper stores the encoding schema, imputation values, selected
encoded columns, nonlinear operations, coefficients, and source-population
means and scales. It can transform compatible new `X` rows using the frozen
definitions. It is not a public DGP component.

Nonlinear score terms may include sparse linear projections, squares, pairwise
products, sine transformations, and threshold indicators. Zero-variance scores
use scale one. Multiple confounding scores are generated by default so distinct
treatment components in future subclasses can depend on distinct shared
causes.

The built-in baseline risk is

\[
\mu_0(x)=f_C\{q_C(x)\}+f_P\{q_P(x)\}.
\]

The assignment predictor uses the same outcome-predictive confounding
contribution, rather than the full baseline risk:

\[
\eta_A(x)
=\alpha
+\lambda B_C q_C(x)
+B_Zq_Z(x).
\]

The frozen `B_C q_C` term is constructed to align with `f_C(q_C)`. This creates
targeted selection while retaining `q_P` as outcome-only. All shared causes
remain functions of released `X` in the observed-confounding reference DGPs.

## Categorical Reference DGP

The first categorical implementation is
`CategoricalSemiSyntheticDataset`. Its primary configuration is:

```python
CategoricalSemiSyntheticDataset(
    real_dataset,
    n_treatments=2,
    target_probabilities=None,
    confounding_strength=1.0,
    target_confounding_bias=None,
    treatment_only_strength=1.0,
    randomized_weight=0.1,
    outcome_effect_scale=1.0,
    effect_heterogeneity_scale=0.5,
    outcome_noise_scale=1.0,
    random_state=42,
)
```

Treatment levels are `0, ..., n_treatments - 1`. When target probabilities are
omitted, they are uniform. The confounded probabilities are

\[
\widetilde p_k(x)
=\operatorname{softmax}_k
\left[\alpha_k+\lambda B_{C,k}q_C(x)+B_{Z,k}q_Z(x)\right].
\]

The exact assignment law is

\[
p_k(x)=(1-\omega)\widetilde p_k(x)+\omega\pi_k,
\]

where `randomized_weight` is \(\omega\) and \(\pi\) is the target randomized
distribution. Intercepts are calibrated numerically so the empirical mean of
the final probabilities matches `target_probabilities`. At \(\omega=1\), the
law is randomized and independent of `X`.

`treatment_only_strength` scales the \(B_Zq_Z\) contribution independently of
confounding strength. It can therefore create treatment predictability without
changing which outcome-predictive scores enter assignment.

`_sample_treatment` draws from these row-specific probabilities. `_log_prob`
returns the logarithm of the selected category probability. Unsupported but
well-formed category values receive `-inf`; malformed treatment inputs raise a
validation error.

The treatment basis uses the first category as reference:

\[
b(a)=\left[\mathbf 1(a=1),\ldots,
\mathbf 1(a=K-1)\right]^\top.
\]

## Scalar-Continuous Reference DGP

The first continuous implementation is
`ContinuousSemiSyntheticDataset`. Its primary configuration is:

```python
ContinuousSemiSyntheticDataset(
    real_dataset,
    treatment_domain="real",
    confounding_strength=1.0,
    target_confounding_bias=None,
    treatment_only_strength=1.0,
    randomized_weight=0.1,
    treatment_noise_scale=1.0,
    outcome_effect_scale=1.0,
    effect_heterogeneity_scale=0.5,
    outcome_noise_scale=1.0,
    random_state=42,
)
```

The confounded latent treatment is Gaussian:

\[
Z\mid X=x\sim N\!\left(
\alpha+\lambda b_C^\top q_C(x)+b_Z^\top q_Z(x),
\sigma_A^2
\right).
\]

`treatment_only_strength` scales the \(b_Z^\top q_Z\) contribution independently
of \(\lambda\). With probability \(\omega\), a latent treatment is instead drawn
from \(N(0,\sigma_A^2)\), independently of `X`. Both mixture components use the
same configured transformation:

\[
A=T(Z)=
\begin{cases}
Z, & \text{real},\\
\exp(Z), & \text{positive},\\
L+(U-L)\operatorname{logistic}(Z), & \text{bounded }[L,U].
\end{cases}
\]

`treatment_domain` accepts `"real"`, `"positive"`, or a finite `(lower, upper)`
pair. `_log_prob` uses a stable log-mixture calculation and the exact
change-of-variables Jacobian. It returns `-inf` outside the configured support.
Treatments are never clipped.

The default continuous treatment basis contains standardized linear,
quadratic, sinusoidal, and hinge terms over a frozen central intervention
region. That region is obtained by transforming fixed lower and upper quantiles
of the randomized latent Gaussian; it is finite even when mathematical support
is unbounded. `get_grid` uses this same region. The implementation freezes every
centering, scaling, quantile, and knot quantity during structural preparation.

## Unified Gaussian Outcome Surface

Both concrete DGPs implement

\[
m(a,x)=
\mu_0(x)
+b(a)^\top\beta
+b(a)^\top\Gamma q_M(x).
\]

`outcome_effect_scale` controls the scale of the frozen main-effect
coefficients and `effect_heterogeneity_scale` controls the interaction
coefficients. The reference treatment has a zero treatment basis, making
`mu_0` its conditional mean.

Observed outcomes are

\[
Y=m(A,X)+\sigma_Y\epsilon,
\qquad \epsilon\sim N(0,1),
\]

where `outcome_noise_scale` is \(\sigma_Y\). `_predict_y` never includes this
noise.

The outcome implementation is kept as private numerical functions shared by
the two concrete classes, not as an `OutcomeDGP` object. A future Bernoulli or
count DGP overrides `_sample_outcome` and applies the appropriate inverse link
inside `_predict_y` while retaining the same base-class contract.

## Confounding Calibration

`confounding_strength` directly supplies \(\lambda\) when
`target_confounding_bias` is `None`. When a target is provided, the structural
scores and outcome surface remain fixed while preparation varies only
\(\lambda\), recalibrating categorical intercepts at each candidate value.

Calibration is private implementation machinery, not a public diagnostics or
calibrator object. It uses the exact oracle functions to calculate

\[
\mu_{\mathrm{do}}(a)=\frac1n\sum_i m(a,X_i)
\]

and

\[
\mu_{\mathrm{obs}}(a)=
\frac{\sum_i g(a\mid X_i)m(a,X_i)}
{\sum_i g(a\mid X_i)}.
\]

The intervention-specific distortion is

\[
\Delta_{\mathrm{conf}}(a)
=\mu_{\mathrm{obs}}(a)-\mu_{\mathrm{do}}(a).
\]

The normalized calibration target is measured relative to

\[
s_0=\operatorname{sd}\{\mu_0(X_i)\}.
\]

For categorical treatments, the scalar bias summary is

\[
\frac{
\max_k\Delta_{\mathrm{conf}}(k)
-\min_k\Delta_{\mathrm{conf}}(k)
}{s_0},
\]

which is the maximum absolute pairwise naive-contrast distortion divided by
baseline-risk standard deviation.

For continuous treatments, the scalar summary is

\[
\frac1{s_0}
\left\{
\frac1M\sum_{m=1}^M
\Delta_{\mathrm{conf}}(a^{(m)})^2
\right\}^{1/2},
\]

using the deterministic intervention grid returned by `get_grid`.

`target_confounding_bias=0.5` therefore requests distortion equal to half a
baseline-risk standard deviation. Preparation stores the achieved values as
`confounding_strength_`, `confounding_bias_` in raw outcome units, and
`confounding_bias_ratio_`.

The numerical search brackets a crossing while holding all non-assignment
structure fixed, then refines it to a fixed internal tolerance. If the target
is below the bias at the minimum strength, above the attainable bias under the
configured randomized mixture/noise, or normalization is impossible because
`s_0` is zero, preparation raises an informative `ValueError` containing the
attainable range.

Overlap remains a separate property. `randomized_weight` and, for continuous
treatment, `treatment_noise_scale` control it. The base API does not add a
public `oracle_diagnostics` method: callers can derive any desired diagnostic
from `predict_y` and `log_prob`, and broader reporting belongs in benchmark
evaluation code.

## Input, Output, and Validation

The base validates that `real_dataset` is a `BaseDataset`, is cloneable, and
returns a nonempty covariate frame with unique columns. It derives `n` from
that frame. Source treatment and outcome values are not inspected by the
built-in DGPs.

The private encoder:

- imputes missing numeric values with frozen source-population medians;
- imputes missing categorical values with a frozen sentinel/category;
- standardizes numeric features;
- encodes categorical features with unknown-category handling;
- rejects infinite numeric values; and
- preserves the released source frame without applying these transformations
  to it.

A numeric source column with no finite observed value cannot define an
imputation value and raises a validation error.

Oracle calls accept NumPy, pandas, and Polars inputs through the same coercion
pattern as `BaseSyntheticDataset`. `X` and treatment inputs must have equal row
counts. New `X` frames must contain the source feature columns in the expected
schema; extra or missing columns receive a clear validation error rather than
being silently reordered or discarded.

Concrete DGPs validate category counts, probability vectors, domain bounds,
nonnegative confounding settings, positive noise scales, and
`randomized_weight` in `[0, 1]`. Treatment outputs use their declared dynamic
`column_types` so the categorical and continuous DGPs participate correctly in
the existing datatype and discovery machinery.

## Files and Exports

The implementation is split into three focused modules:

```text
src/skcausal/datasets/
    semi_synthetic.py
    semi_synthetic_categorical.py
    semi_synthetic_continuous.py
```

`semi_synthetic.py` contains `BaseSemiSyntheticDataset` and private shared
numerical helpers. The other modules contain the two complete reference DGPs.
`src/skcausal/datasets/__init__.py` exports all three public classes.

No existing semi-synthetic class is renamed or changed beyond any narrowly
necessary shared compatibility fix discovered during implementation.

## Testing

Tests must cover:

1. Exact preservation of source `X`, including columns and row order.
2. Cloning the source dataset without mutating it.
3. NumPy, pandas, and Polars oracle inputs.
4. Structural reproducibility from constructor seeds.
5. Replication reproducibility and non-mutation of `load()` outputs.
6. Categorical probability normalization and target marginal calibration.
7. Categorical randomized assignment at `randomized_weight=1`.
8. Continuous density normalization for real, positive, and bounded domains.
9. Exact mixture log-probabilities and support behavior.
10. Agreement between `predict_y` and the generated Gaussian conditional mean.
11. Confounding-target calibration within numerical tolerance.
12. Clear failures for invalid source data, parameters, schemas, and treatments.
13. Dataset discovery, cloning, parameter inspection, and `get_test_params`.
14. The existing full test suite.

The randomized sanity test verifies that fully randomized assignment produces
zero oracle confounding distortion up to numerical tolerance. Calibration tests
verify that score definitions and outcome coefficients remain fixed while
assignment strength changes.

## Extension Boundaries

The base signatures already accept treatment frames with any number and mix of
columns. A future vector or mixed treatment DGP implements a joint sampler and
joint `_log_prob`; no base change is required. Ordinal and count DGPs likewise
only provide concrete assignment laws.

Alternative outcome families override the mean/link and sampling hooks. Hidden
confounding can be represented by frozen internal unit-level scores consumed by
both oracle functions but omitted from the released `X`; such a subclass must
clearly label its identification violation and define whether predictions on
new rows are supported.

A source-outcome prognostic model can be added in a subclass's structural
preparation or through a future private helper. The base does not require source
outcomes and therefore does not couple real-dataset acquisition to a particular
prediction model.

Benchmark condition matrices, artifact persistence, intervention-support
reports, and performance summaries remain outside the dataset object. They can
use the stable `load`, `sample`, `predict_y`, `predict_curve`, and `log_prob`
interfaces without changing the generation core.
