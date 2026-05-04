# skcausal

`skcausal` is a machine learning library for estimating average causal responses from observational data. It uses one consistent `fit(X, t, y)` contract across categorical, continuous, and multi-column treatments, while letting you plug in familiar scikit-learn models for nuisance regression, density estimation, and final smoothing.

It is built around average potential outcomes and dose-response functions rather than per-unit counterfactual prediction.

[Documentation](https://skcausal.com) | [Continuous treatments](https://skcausal.com/latest/examples/continuous_treatments.html) | [Categorical treatments](https://skcausal.com/latest/examples/categorical_treatments.html) | [Multidimensional treatments](https://skcausal.com/latest/examples/multidimensional_treatments.html)

## What The Package Includes

- Average causal response estimators for direct, GPS/IPW-style, and doubly robust workflows
- One public interface for categorical, continuous, and multidimensional treatment tables
- Treatment density estimators and density pipelines for nuisance modeling
- Synthetic and semi-synthetic datasets with known truth for benchmarking
- Discovery utilities for listing datasets, density estimators, and causal estimators
- Optional plotting helpers for comparing estimated curves against observed data or ground truth

## Installation

Install the base package:

```bash
pip install skcausal
```

Optional extras:

```bash
pip install "skcausal[plotting]"
pip install "skcausal[torch]"
pip install "skcausal[optuna]"
pip install "skcausal[skpro]"
```

## Core Contract

Average-response estimators follow the same public workflow:

```python
estimator.fit(X, t, y)
curve = estimator.predict(requested_treatments)
```

Where:

- `X` contains pre-treatment covariates
- `t` contains the observed treatment table
- `y` contains the observed outcome table
- `requested_treatments` contains the intervention levels or grid you want to evaluate

For categorical treatments, `requested_treatments` is usually a short table of valid levels. For continuous treatments, it is usually a dense treatment grid. Predictions return one average response per requested row, averaging over the covariate sample stored during `fit`.

## Quick Start

The example below fits a continuous-treatment direct regression estimator on a synthetic dataset with known ground truth.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from skcausal.causal_estimators import DirectRegressor
from skcausal.datasets import SyntheticDataset2
from skcausal.utils.treatment_grid import make_cartesian_treatment_grid

dataset = SyntheticDataset2(n=2000, n_features=6, random_state=0)
X, t, y = dataset.load()

estimator = DirectRegressor(
	outcome_regressor=RandomForestRegressor(
		n_estimators=200,
		min_samples_leaf=5,
		random_state=0,
	)
)

estimator.fit(X, t, y)

grid = make_cartesian_treatment_grid(t, n_continuous_points=25)
estimated_curve = estimator.predict(grid)
truth_curve = np.asarray(dataset.predict(X, grid)).reshape(-1)
```

This pattern generalizes across treatment types:

- Fit on aligned `X`, `t`, and `y`
- Create a treatment table that represents the interventions you want to query
- Call `predict(...)` on that treatment table
- On synthetic datasets, compare against `dataset.predict(X, requested_treatments)` when ground truth is available

## Choosing A Starting Estimator

| Setting                     | Good starting points                                                                          |
| --------------------------- | --------------------------------------------------------------------------------------------- |
| Categorical treatments      | `CategoricalDoublyRobust`, `CategoricalDirectMethod`, `CategoricalInversePropensityWeighting` |
| Continuous treatments       | `DirectRegressor`, `GPS`, `DoublyRobustPseudoOutcome`                                         |
| Multidimensional treatments | `DirectRegressor`, `GPS`, `DoublyRobustPseudoOutcome`                                         |
| Observational baseline only | `DirectNoCovariates`                                                                          |

`DirectNoCovariates` is useful as a sanity-check baseline, but it does not adjust for confounding.

## Discover Available Components

You can inspect the registered public objects programmatically:

```python
from skcausal.utils.lookup import (
	all_causal_average_response_estimators,
	all_datasets,
	all_density_estimators,
)

all_causal_average_response_estimators(
	as_dataframe=True,
	filter_tags={"capability:t_type": "continuous"},
	return_tags=["capability:t_type", "capability:multidimensional_treatment"],
)
```

This is useful when you want to discover which estimators support a particular treatment type or whether an estimator handles multidimensional treatments.

## Main Public Modules

- `skcausal.causal_estimators`: average-response estimators and estimator pipelines
- `skcausal.density`: treatment density estimators and density pipelines
- `skcausal.datasets`: synthetic and semi-synthetic benchmark datasets
- `skcausal.plotting`: optional plotting helpers for response curves and comparisons
- `skcausal.utils.lookup`: object discovery helpers built on top of `skbase.lookup`

## Documentation

- [Getting started](https://skcausal.com/book/getting-started.html)
- [Estimator contract](https://skcausal.com/book/estimator-contract.html)
- [Continuous-treatment workflow](https://skcausal.com/examples/continuous_treatments.html)
- [Categorical-treatment workflow](https://skcausal.com/examples/categorical_treatments.html)
- [Multidimensional-treatment workflow](https://skcausal.com/examples/multidimensional_treatments.html)
- [Dataset catalog](https://skcausal.com/research/dataset_catalog.html)
- [Benchmarking notes](https://skcausal.com/research/benchmarking.html)

## Scope

`skcausal` is designed around average-response estimation from observational data:

- Average potential outcomes for binary or categorical treatments
- Average dose-response functions for continuous treatments
- Average response surfaces for multidimensional treatments

It does not present `predict(X_new)` as the main API. Instead, prediction is framed around requested intervention tables, and estimators average over the covariate sample seen at fit time.