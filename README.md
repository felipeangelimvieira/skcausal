![skcausal](docs/logo.svg)
# skcausal: A Machine Learning Library for Causal Inference


## Overview

`skcausal` is a Python library designed to provide machine learning tools for causal inference. It allows users to estimate causal effects using techniques such as propensity score weighting, generalized propensity scores, and optimal hyperparameter tuning. Built on top of `polars`, `optuna`, and `pytorch-lightning`, `skcausal` offers scalable and flexible implementations of state-of-the-art causal response estimation methods.

## Features

- **Causal Estimation Models:** Implements various approaches to causal effect estimation, including direct response modeling, propensity weighting, and GPS-based methods.
- **Hyperparameter Optimization:** Integrates `optuna` for tuning causal models efficiently.
- **Propensity Score Weighting:** Supports several weighting techniques, including synthetic classifier-based estimators and neural network-based density ratio estimation.
- **Flexible Treatment Modeling:** Supports both binary and continuous treatment variables, as well as multi-dimensional treatment estimation.
- **Seamless Integration with Machine Learning Pipelines:** Provides compatibility with `sklearn`-like API and supports modern ML techniques for regression and classification.

## Installation

To install `skcausal`, use:
```bash
pip install skcausal
```

## Modules

### 1. Causal Estimators (`skcausal.causal_estimators`)
This module provides the core classes for estimating the average dose-response function (ADRF) and individual treatment effects (ITE).

#### Key Classes:
- `BaseCausalResponseEstimator` - Abstract base class for all causal estimators.
- `GPS` - Implements the Generalized Propensity Score method.
- `PropensityWeightingDiscrete` - Uses Propensity Score Weighting for discrete treatments.
- `PropensityWeightingContinuous` - Extends Propensity Score Weighting for continuous treatments.
- `BinaryDoublyRobust` - Combines outcome regression and propensity weighting for doubly robust estimation.

### 2. Hyperparameter Tuning (`skcausal.tuning`)
Provides tools for hyperparameter tuning using `optuna`.

#### Key Classes:
- `OptunaCausalResponseEstimator` - Wraps a causal estimator with an Optuna-based tuning procedure.

### 3. Weight Estimators (`skcausal.weight_estimators`)
Contains methods for estimating balancing weights used in causal inference.

#### Key Classes:
- `BaseBalancingWeightRegressor` - Base class for balancing weight estimation.
- `BinaryClassifierWeightRegressor` - Learns propensity scores via binary classification.
- `DiscriminativeWeightRegressor` - Creates a synthetic classification problem to estimate inverse probability of treatment weights (IPTW).
- `TreatmentDensityRatioRegressor` - Uses a deep learning model to estimate density ratio weights.
- `InterpolateNeuralWeightRegressor` - A neural network-based method for weight estimation with linear interpolation.

### 4. Polars Utility Functions (`skcausal.polars`)
Helper functions for data preprocessing, including:
- `convert_categorical_to_dummies()` - Converts categorical features to dummy variables.
- `to_dummies()` - One-hot encodes categorical features.
- `assert_schema_equal()` - Ensures consistency between data schemas.

## Example Usage

### Causal Inference with Generalized Propensity Score (GPS)
```python
import polars as pl
import numpy as np
from skcausal.causal_estimators import GPS
from skcausal.weight_estimators import BinaryClassifierWeightRegressor
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data
n_samples = 1000
X = np.random.rand(n_samples, 5)
t = np.random.choice([0, 1], size=n_samples)
y = 2 * t + np.random.randn(n_samples)

X_df = pl.DataFrame(X, schema=[f"x{i}" for i in range(X.shape[1])])
t_df = pl.DataFrame({"treatment": t})
y_df = pl.DataFrame({"outcome": y})

# Define weight estimator
treatment_regressor = BinaryClassifierWeightRegressor(RandomForestClassifier())

# Define GPS estimator
gps_estimator = GPS(treatment_regressor, outcome_regressor=RandomForestClassifier())
gps_estimator.fit(X_df, y_df, t_df)

# Predict treatment effect
ate = gps_estimator.predict_average_treatment_effect(X_df, t_df)
print("Estimated ATE:", ate)
```

### Hyperparameter Tuning with Optuna
```python
from skcausal.tuning import OptunaCausalResponseEstimator
from causal_experiment.evaluation.metrics.dose_response import EMSE
from causal_experiment.datasets.synthetic_wang import SyntheticBidimensionalDataset
from skcausal.causal_estimators.direct_dynamicnet import DirectDynamicNet
from optuna.distributions import IntUniformDistribution, LogUniformDistribution

# Define model and dataset
model = DirectDynamicNet(n_epochs=10)
metric = EMSE()
dataset = SyntheticBidimensionalDataset().prepare(n=1000)
param_grid = {
    "learning_rate": LogUniformDistribution(1e-5, 1e-1),
    "batch_size": IntUniformDistribution(32, 512),
}

# Run hyperparameter tuning
optuna_estimator = OptunaCausalResponseEstimator(model, metric, param_grid, dataset, n_evals=10)
optuna_estimator.tune()
print("Best Parameters:", optuna_estimator.best_params_)
```

## Contributing
We welcome contributions to `skcausal`! If you want to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Write tests for your code.
4. Submit a pull request.

## License
`skcausal` is licensed under the MIT License. See the `LICENSE` file for details.

