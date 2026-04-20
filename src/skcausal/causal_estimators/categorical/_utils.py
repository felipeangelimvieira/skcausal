import numpy as np
import pandas as pd

from skcausal.causal_estimators._density_utils import is_stabilized_density


def _normalize_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def coerce_categorical_treatment(
    t: pd.DataFrame,
    *,
    estimator_name: str,
    argument_name: str = "t",
) -> np.ndarray:
    """
    Return categorical treatment rows as hashable scalar or tuple keys.

    Coerce the treatment to numpy array of tuples or scalars.
    Estimator name and argument name are used for error messages.

    Parameters
    ----------
    t : pd.DataFrame
        Treatment DataFrame to coerce.
    estimator_name : str
        Name of the estimator for error messages.
    argument_name : str, default="t"
        Name of the argument for error messages.

    Returns
    -------
    np.ndarray
        Array of hashable treatment values, either as scalars (for single-column
        treatments) or tuples (for multi-column treatments).
    """

    if t.shape[1] == 0:
        raise ValueError(
            f"{estimator_name} requires {argument_name} to contain at least one treatment column."
        )

    normalized_columns = []
    for column_name in t.columns:
        series = t[column_name]
        if series.isna().any():
            raise ValueError(
                f"{estimator_name} requires {argument_name} to contain no missing values."
            )
        normalized_columns.append(
            [_normalize_scalar(value) for value in series.to_list()]
        )

    if t.shape[1] == 1:
        # If a single treatment column we take the single normalized column
        return np.asarray(normalized_columns[0], dtype=object)

    joint_values = np.empty(len(t), dtype=object)
    for index, row_values in enumerate(zip(*normalized_columns)):
        joint_values[index] = tuple(row_values)
    return joint_values


def get_treatment_levels(treatment_values: np.ndarray) -> list:
    """Return observed treatment levels in order of first appearance."""

    return pd.Series(treatment_values, dtype=object).drop_duplicates().tolist()


def treatment_value_mask(treatment_values: np.ndarray, level) -> np.ndarray:
    """Return an elementwise equality mask for scalar or tuple treatment levels."""

    return np.fromiter((value == level for value in treatment_values), dtype=bool)


def validate_requested_treatment_values(
    requested_values: np.ndarray,
    *,
    observed_levels: list,
    estimator_name: str,
    argument_name: str = "t",
) -> None:
    missing_levels = [
        value for value in requested_values if value not in observed_levels
    ]
    if missing_levels:
        raise ValueError(
            f"{estimator_name} received unseen treatment values in {argument_name}: "
            f"{missing_levels!r}. Expected values drawn from {observed_levels!r}."
        )


def constant_treatment_frame(
    reference_t: pd.DataFrame,
    *,
    value,
    n_rows: int,
) -> pd.DataFrame:
    """Create a constant treatment frame matching the reference columns."""

    if reference_t.shape[1] == 1:
        return pd.DataFrame({reference_t.columns[0]: [value] * n_rows})

    if not isinstance(value, tuple) or len(value) != reference_t.shape[1]:
        raise ValueError(
            "Expected a tuple of treatment values matching the reference columns "
            f"{reference_t.columns.tolist()}, but received {value!r}."
        )

    return pd.DataFrame(
        {
            column_name: [column_value] * n_rows
            for column_name, column_value in zip(reference_t.columns, value)
        }
    )


def density_to_probability_matrix(
    density_estimator,
    density_matrix: np.ndarray,
    marginals: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert conditional or stabilized density outputs into probabilities."""

    probabilities = np.asarray(density_matrix, dtype=float)
    if is_stabilized_density(density_estimator):
        probabilities = probabilities * np.asarray(marginals, dtype=float).reshape(
            1, -1
        )

    denom = np.clip(probabilities.sum(axis=1, keepdims=True), eps, None)
    return probabilities / denom
