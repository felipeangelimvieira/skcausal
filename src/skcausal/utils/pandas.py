import numpy as np
import pandas as pd
from pandas.api import types as pdt

INTEGER_DTYPES = [
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
]

FLOAT_DTYPES = [np.dtype("float32"), np.dtype("float64")]

BOOLEAN_DTYPES = [np.dtype("bool"), "bool", "boolean"]

ALL_DTYPES = ["category", *BOOLEAN_DTYPES, *INTEGER_DTYPES, *FLOAT_DTYPES]


def _dtype_equals(left, right) -> bool:
    try:
        return np.dtype(left) == np.dtype(right)
    except TypeError:
        return left == right


def is_dtype_supported(dtype, allowed_dtypes) -> bool:
    """Return True if ``dtype`` is compatible with any entry in ``allowed_dtypes``."""

    for allowed in allowed_dtypes:
        if allowed == "category":
            if pdt.is_categorical_dtype(dtype):
                return True
            continue
        if allowed in {"bool", "boolean"}:
            if pdt.is_bool_dtype(dtype):
                return True
            continue
        if allowed == "numeric":
            if pdt.is_numeric_dtype(dtype):
                return True
            continue
        if isinstance(allowed, np.dtype):
            if not pdt.is_extension_array_dtype(dtype) and _dtype_equals(dtype, allowed):
                return True
            continue
        if isinstance(allowed, type):
            if isinstance(dtype, allowed):
                return True
            continue
        if dtype == allowed:
            return True
    return False


def assert_schema_equal(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    if list(df_a.columns) != list(df_b.columns):
        raise AssertionError(
            "DataFrames do not share the same column order: "
            f"{df_a.columns.tolist()} != {df_b.columns.tolist()}"
        )
    if not df_a.dtypes.equals(df_b.dtypes):
        raise AssertionError(
            "DataFrames do not share the same dtypes: "
            f"{df_a.dtypes.to_dict()} != {df_b.dtypes.to_dict()}"
        )


def _ensure_categorical(series: pd.Series) -> pd.Series:
    if not pdt.is_categorical_dtype(series):
        return series.astype("category")
    return series


def to_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    series = _ensure_categorical(df[column])
    categories = list(series.cat.categories)
    df_no_col = df.drop(columns=[column])

    dummy_series = []
    for cat in categories[:-1]:
        dummy = (series == cat).astype(np.uint8)
        dummy.name = f"{column}__dummy_{cat}"
        dummy_series.append(dummy)

    if dummy_series:
        return pd.concat([df_no_col, *dummy_series], axis=1)
    return df_no_col


def convert_categorical_to_dummies(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in list(result.columns):
        if pdt.is_categorical_dtype(result[col]):
            result = to_dummies(result, col)
    return result
