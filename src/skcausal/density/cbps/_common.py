"""Helpers shared by the CBPS estimator classes."""

from __future__ import annotations

import numpy as np
import pandas as pd

VALID_METHODS = ("over", "exact")


def validate_method(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method {method!r}. Expected one of {list(VALID_METHODS)}."
        )


def covariates_to_array(X: pd.DataFrame) -> np.ndarray:
    """Convert a covariate frame to a float matrix, rejecting non-numeric columns."""
    non_numeric = [
        column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if non_numeric:
        raise ValueError(
            "CBPS requires numeric covariates. Non-numeric columns: "
            f"{non_numeric}. Encode them before fitting."
        )
    return X.to_numpy(dtype=float)
