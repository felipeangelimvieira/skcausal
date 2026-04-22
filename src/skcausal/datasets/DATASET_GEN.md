# Data-Generating Process for Causal Benchmark Experiments

This document specifies the data-generating process (DGP) used to produce semi-synthetic causal datasets for experimental evaluation. The DGP is designed so that every quantity needed for evaluation—conditional means, treatment densities, and counterfactual surfaces—is analytically known.

---

## 1. Structural Model

The causal graph contains exactly three node types:

```
X ──→ T
X ──→ Y
T ──→ Y
```

All confounding is observed. There is no hidden common cause of $T$ and $Y$.

The structural equations are:

$$
X \sim P_X
$$

$$
T \mid X \sim F_T(\cdot \mid X)
$$

$$
Y = \mu(X, T) + \varepsilon_Y, \quad \varepsilon_Y \sim \mathcal{N}(0, \sigma_Y^2)
$$

with $\varepsilon_Y \perp\!\!\!\perp (X, T)$.

The conditional mean decomposes additively:

$$
\mu(X, T) = g(X) + \tau(X, T)
$$

where $g(X)$ is the prognostic baseline (the part of $Y$ explained by $X$ alone) and $\tau(X, T)$ is the causal effect of treatment.

---

## 2. Assumptions

The DGP is constructed to satisfy:

1. **No unmeasured confounding (ignorability).** $(Y^{(t)} \perp\!\!\!\perp T) \mid X$ for all $t$.
2. **Positivity.** $f_T(t \mid X = x) > 0$ for all $(x, t)$ in the support of interest.
3. **Consistency.** $Y = Y^{(T)}$.
4. **Known $\mu(X, T)$.** The conditional mean $E[Y \mid X, T]$ is a closed-form function.
5. **Known $f_T(t \mid x)$.** The treatment density (or mass function) is a closed-form function.

Assumptions 4 and 5 are what make the dataset a usable benchmark: any estimator's output can be compared to the ground truth for both ADRF estimation and density/weight estimation.

---

## 3. Covariates

### 3.1 Source

Covariates are drawn from a real regression dataset $(X_{\text{real}}, y_{\text{real}})$. Only $X_{\text{real}}$ is used; $y_{\text{real}}$ is discarded.

### 3.2 Sampling

Rows are drawn with replacement (bootstrap) from $X_{\text{real}}$:

$$
X_i \overset{\text{iid}}{\sim} \hat{P}_X \quad \text{(empirical distribution of } X_{\text{real}}\text{)}
$$

### 3.3 Preprocessing

Before entering the structural equations, covariates are preprocessed into a numeric matrix:

1. Standardize all continuous columns to zero mean, unit variance.
2. Encode categoricals via ordinal encoding (not one-hot) to keep dimensionality bounded.
3. Store the fitted scaler and encoder as frozen artifacts for reproducibility.

Let $p$ denote the number of columns after preprocessing.

### 3.4 Feature Map

A deterministic feature map $\phi: \mathbb{R}^p \to \mathbb{R}^d$ enriches the raw covariates for use in the treatment score and outcome function. The default map is:

$$
\phi(X) = \bigl[\, \tilde{X},\; \tilde{X}^2,\; \sin(\tilde{X}) \,\bigr]
$$

where $\tilde{X}$ is the standardized numeric matrix and all operations are element-wise. This gives $d = 3p$ features.

When $p > 20$, reduce dimensionality first: retain the top $k = \min(p, 20)$ columns by marginal variance, then apply $\phi$ to those $k$ columns. This avoids a combinatorial explosion while preserving enough structure for nonlinear confounding.

The same $\phi$ feeds both treatment and outcome. This overlap is what creates confounding.

---

## 4. Treatment Assignment

### 4.1 Treatment Score

Define a latent treatment propensity score:

$$
s(X) = \phi(X)^\top \beta_T
$$

where $\beta_T \in \mathbb{R}^d$ is a fixed coefficient vector. The coefficients are drawn once at dataset-creation time from $\mathcal{N}(0, 1)$ and frozen.

### 4.2 Continuous Treatment on $[0, 1]$

Map the score to a mean parameter:

$$
m(X) = \sigma(s(X))
$$

where $\sigma$ is the logistic function. Then sample:

$$
T \mid X \sim \text{Beta}\bigl(\kappa \, m(X),\; \kappa \, (1 - m(X))\bigr)
$$

The concentration parameter $\kappa > 0$ controls overlap:

| $\kappa$                | Behavior                                                     |
| ----------------------- | ------------------------------------------------------------ |
| $\kappa \leq 2$         | Very dispersed; weak confounding                             |
| $5 \leq \kappa \leq 30$ | Moderate confounding; good overlap                           |
| $\kappa \geq 50$        | Strong confounding; treatment nearly deterministic given $X$ |

**Default:** $\kappa = 20$.

**Positivity audit.** After sampling, verify that $m(X) \in [\epsilon, 1-\epsilon]$ for all sampled $X$, with $\epsilon = 0.01$. If violated, clip $s(X)$ to $[\sigma^{-1}(\epsilon), \sigma^{-1}(1-\epsilon)]$.

**Density.** The treatment density is:

$$
f_T(t \mid X) = \frac{t^{\kappa m(X) - 1}(1-t)^{\kappa(1-m(X))-1}}{B\bigl(\kappa m(X),\; \kappa(1-m(X))\bigr)}
$$

This is the Beta PDF and is available in closed form via `scipy.stats.beta`.

### 4.3 Binary Treatment

$$
p(X) = \text{clip}\bigl(\sigma(s(X)),\; 0.05,\; 0.95\bigr)
$$

$$
T \mid X \sim \text{Bernoulli}(p(X))
$$

Clipping ensures overlap by construction.

### 4.4 Discrete Treatment ($K$ Levels)

Generate a latent continuous treatment $T^*$ as in Section 4.2, then discretize using quantile thresholds:

$$
T = k \quad \text{iff} \quad q_{k-1} < T^* \leq q_k
$$

where $q_0 = 0$, $q_K = 1$, and the interior thresholds $q_1, \ldots, q_{K-1}$ are the population quantiles of $T^*$ (computable from the Beta CDF). Quantile thresholds ensure balanced class sizes.

---

## 5. Outcome Function

### 5.1 Prognostic Baseline

$$
g(X) = \phi(X)^\top \beta_Y
$$

with $\beta_Y \in \mathbb{R}^d$ drawn once from $\mathcal{N}(0, 1)$ and frozen. The coefficients $\beta_Y$ must share support with $\beta_T$ (both use the same $\phi$) but must not be identical—this ensures confounding exists but the outcome is not a deterministic function of the treatment score alone.

### 5.2 Treatment Effect (Single Treatment)

The causal effect $\tau(X, T)$ is defined in one of the following tiers, ordered by complexity:

**Tier 1 — Homogeneous polynomial:**

$$
\tau(X, T) = \alpha_1 T + \alpha_2 T^2 + \alpha_3 T^3
$$

**Tier 2 — Heterogeneous polynomial:**

$$
\tau(X, T) = (\alpha_1 T + \alpha_2 T^2) \cdot h(X)
$$

where $h(X) = \phi(X)^\top \beta_H$ with $\beta_H \in \mathbb{R}^d$ drawn and frozen.

**Tier 3 — Heterogeneous with interaction:**

$$
\tau(X, T) = \alpha_1 T + \alpha_2 T^2 + \alpha_3 T \cdot h(X)
$$

**Default for experiments:** Tier 3, with $\alpha_1 = 2.0$, $\alpha_2 = -1.5$, $\alpha_3 = 0.8$.

### 5.3 Outcome Noise

$$
Y = \mu(X, T) + \varepsilon_Y, \quad \varepsilon_Y \sim \mathcal{N}(0, \sigma_Y^2)
$$

**Default:** $\sigma_Y^2 = 0.25$.

To support heteroscedastic experiments, an optional variant is:

$$
\varepsilon_Y \sim \mathcal{N}\bigl(0,\; \sigma_Y^2 \cdot (1 + |g(X)|)\bigr)
$$

This increases noise where the baseline signal is large, making the estimation problem harder in high-prognostic regions.

### 5.4 Signal-to-Noise Calibration

After drawing $\beta_Y$, $\beta_T$, and the $\alpha$ coefficients, compute the empirical signal-to-noise ratio on a calibration sample of size $n_{\text{cal}} = 10\,000$:

$$
\text{SNR}_g = \frac{\text{Var}(g(X))}{\sigma_Y^2}, \quad \text{SNR}_\tau = \frac{\text{Var}(\tau(X, T))}{\sigma_Y^2}
$$

If $\text{SNR}_\tau < 0.1$ or $\text{SNR}_\tau > 100$, rescale the $\alpha$ coefficients proportionally so that $\text{SNR}_\tau \in [0.5, 10]$. Report both SNR values with every generated dataset.

---

## 6. Multiple Treatments

### 6.1 Treatment Vector

Let $T = (T_1, \ldots, T_K)$. Each component has its own score:

$$
s_j(X) = \phi(X)^\top \beta_{T_j}
$$

and is sampled independently conditional on $X$:

$$
T_j \mid X \sim \text{Beta}\bigl(\kappa_j \, m_j(X),\; \kappa_j \, (1 - m_j(X))\bigr)
$$

Treatments are correlated through shared dependence on $X$, not through direct dependence on each other. This preserves the DAG structure (no edges between treatment nodes) while still producing empirically correlated treatments.

### 6.2 Joint Outcome Surface

$$
\mu(X, T) = g(X) + \sum_{j=1}^{K} \tau_j(X, T_j) + \sum_{j < k} \rho_{jk}\, T_j\, T_k
$$

Each $\tau_j$ follows the same tier structure from Section 5.2. The interaction terms $\rho_{jk} T_j T_k$ are optional and controlled by a parameter flag.

### 6.3 Joint Density

Because treatments are conditionally independent given $X$:

$$
f_T(t_1, \ldots, t_K \mid X) = \prod_{j=1}^{K} f_{T_j}(t_j \mid X)
$$

Each factor is a Beta density. The joint density is therefore closed-form.

---

## 7. Evaluation Targets

The DGP provides ground truth for the following quantities.

### 7.1 Average Dose-Response Function (ADRF)

$$
\text{ADRF}(t) = E_X[\mu(X, t)] = E_X[g(X) + \tau(X, t)]
$$

Estimated by Monte Carlo over the covariate sample. Evaluation metric: integrated absolute error or pointwise MSE over a treatment grid $\{t_1, \ldots, t_G\}$.

### 7.2 Conditional Average Treatment Effect (CATE)

For binary treatment:

$$
\text{CATE}(x) = \mu(x, 1) - \mu(x, 0)
$$

For continuous treatment at dose $t$ relative to reference $t_0$:

$$
\text{CATE}(x; t, t_0) = \mu(x, t) - \mu(x, t_0)
$$

### 7.3 Treatment Density

$$
f_T(t \mid x) \quad \text{(Section 4.2)}
$$

Ground truth for evaluating density estimators (e.g., `BaseDensityEstimator.predict_density`) and GPS-based causal estimators.

### 7.4 Balancing Weights

For a target distribution $Q(T)$ (typically the marginal $f_T(t)$):

$$
w(X, T) = \frac{Q(T)}{f_T(T \mid X)}
$$

Ground truth for evaluating `BaseBalancingWeightRegressor.predict_sample_weight`.

---

## 8. Mapping to `BaseDataset` API

| DGP Component                   | API Method             | Returns                                      |
| ------------------------------- | ---------------------- | -------------------------------------------- |
| Bootstrap $X$ from $\hat{P}_X$  | `get_covariates(n)`    | $n \times p$ covariate matrix                |
| Sample $T \mid X$               | `get_treatments(X)`    | Treatment array                              |
| $\mu(X, T) = g(X) + \tau(X, T)$ | `predict_y(X, T)`      | Conditional mean (exact)                     |
| $Y = \mu(X, T) + \varepsilon_Y$ | `get_outcomes(X, T)`   | Noisy outcome                                |
| $f_T(t \mid X)$                 | `pdf_treatments(T, X)` | Treatment density (exact)                    |
| Treatment evaluation grid       | `get_grid(n)`          | $n$-point grid on $[0, 1]$ or Cartesian grid |

Every method listed above must be implemented by concrete dataset classes. In particular, `pdf_treatments` must return the exact density—not a `NotImplementedError`—because GPS, doubly-robust, and weight-based estimators all depend on it for benchmarking.

---

## 9. Concrete DGP Specifications

The following are the exact DGPs to be used in experiments.

### 9.1 DGP-C: Continuous Treatment, Single

**Covariates.** $X \in \mathbb{R}^{n \times p}$ bootstrapped from a source regression dataset. Default source: any `sklearn` regression dataset with $p \geq 6$.

**Feature map.** $\phi(X) = [\tilde{X}, \tilde{X}^2, \sin(\tilde{X})]$, giving $d = 3p$.

**Treatment score.** $\beta_T \sim \mathcal{N}(0, I_d)$, frozen.

$$
m(X) = \sigma\!\bigl(\phi(X)^\top \beta_T / \sqrt{d}\bigr)
$$

The division by $\sqrt{d}$ normalizes the score variance to $O(1)$.

**Treatment.**

$$
T \mid X \sim \text{Beta}(20\, m(X),\; 20\,(1 - m(X)))
$$

**Outcome score.** $\beta_Y \sim \mathcal{N}(0, I_d)$, frozen, drawn independently from $\beta_T$.

$$
g(X) = \phi(X)^\top \beta_Y / \sqrt{d}
$$

**Heterogeneity score.** $\beta_H \sim \mathcal{N}(0, I_d)$, frozen.

$$
h(X) = \phi(X)^\top \beta_H / \sqrt{d}
$$

**Treatment effect.**

$$
\tau(X, T) = 2.0\, T - 1.5\, T^2 + 0.8\, T \cdot h(X)
$$

**Outcome.**

$$
Y \sim \mathcal{N}\!\bigl(g(X) + \tau(X, T),\; 0.25\bigr)
$$

**Treatment density.**

$$
f_T(t \mid X) = \text{Beta}(t;\; 20\, m(X),\; 20\,(1 - m(X)))
$$

### 9.2 DGP-B: Binary Treatment

Same covariate and feature-map setup as DGP-C.

**Treatment.**

$$
p(X) = \text{clip}\!\bigl(\sigma(\phi(X)^\top \beta_T / \sqrt{d}),\; 0.05,\; 0.95\bigr)
$$

$$
T \mid X \sim \text{Bernoulli}(p(X))
$$

**Treatment effect.**

$$
\tau(X, T) = \alpha\, T + \gamma\, T \cdot h(X)
$$

with $\alpha = 1.0$, $\gamma = 0.5$.

**Outcome.**

$$
Y \sim \mathcal{N}\!\bigl(g(X) + \tau(X, T),\; 0.25\bigr)
$$

### 9.3 DGP-M: Multi-Treatment (Two Continuous)

Same covariate and feature-map setup as DGP-C.

**Treatment scores.** $\beta_{T_1}, \beta_{T_2} \sim \mathcal{N}(0, I_d)$, drawn independently and frozen.

$$
m_j(X) = \sigma\!\bigl(\phi(X)^\top \beta_{T_j} / \sqrt{d}\bigr), \quad j = 1, 2
$$

**Treatments.**

$$
T_j \mid X \sim \text{Beta}(15\, m_j(X),\; 15\,(1 - m_j(X))), \quad j = 1, 2
$$

**Treatment effect.**

$$
\tau(X, T_1, T_2) = 1.2\, T_1 - 0.8\, T_1^2 + 0.9\, T_2 + 0.5\, T_1 T_2 + 0.6\, T_2 \cdot h(X)
$$

**Outcome.**

$$
Y \sim \mathcal{N}\!\bigl(g(X) + \tau(X, T_1, T_2),\; 0.25\bigr)
$$

**Joint density.**

$$
f_T(t_1, t_2 \mid X) = f_{T_1}(t_1 \mid X) \cdot f_{T_2}(t_2 \mid X)
$$

---

## 10. Experimental Protocol

### 10.1 Sample Sizes

Each DGP is evaluated at $n \in \{500,\; 2000,\; 10\,000\}$ to measure performance across low-, moderate-, and high-data regimes.

### 10.2 Replications

For each $(n, \text{DGP})$ pair, generate $R = 20$ independent datasets using different random seeds. Report mean and standard error of each metric across replications.

### 10.3 Train/Test Split

Each dataset is split 80/20 via `BaseDataset.prepare(n, test_ratio=0.2)`. Estimators are fitted on the training set. All evaluation metrics are computed on the test set.

### 10.4 Evaluation Metrics

**ADRF estimation.** Integrated mean squared error:

$$
\text{IMSE} = \frac{1}{G} \sum_{g=1}^{G} \bigl(\widehat{\text{ADRF}}(t_g) - \text{ADRF}(t_g)\bigr)^2
$$

over a grid of $G = 100$ equally spaced points on $[0, 1]$.

**Density estimation.** Mean integrated squared error of $\hat{f}_T(t \mid X)$ against the true Beta density over a held-out set.

**Weight estimation.** Mean squared error of estimated balancing weights versus oracle weights on the test set.

### 10.5 Difficulty Knobs

The DGP exposes the following parameters to control benchmark difficulty:

| Parameter               | Effect                                                               | Default |
| ----------------------- | -------------------------------------------------------------------- | ------- |
| $\kappa$                | Treatment concentration; higher = stronger confounding, less overlap | 20      |
| $\sigma_Y^2$            | Outcome noise; higher = lower SNR                                    | 0.25    |
| $\alpha$ coefficients   | Treatment effect magnitude; rescaled via SNR calibration             | See §9  |
| $p$                     | Covariate dimension; higher = harder nuisance estimation             | 6       |
| Heteroscedasticity flag | Enables noise proportional to $\|g(X)\|$                             | Off     |

---

## 11. Reproducibility

Every generated dataset is fully determined by:

1. The source regression dataset (identified by name and version).
2. The preprocessing pipeline (frozen scaler + encoder).
3. The frozen coefficient vectors $\beta_T, \beta_Y, \beta_H$ (drawn from an `init_seed`).
4. The sampling seed (passed to `prepare(preparation_seed=...)`).

All four must be recorded with experimental results. The coefficient vectors should be serialized (e.g., as `.npz` files) alongside the code.