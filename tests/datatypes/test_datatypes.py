from datetime import date

import pandas as pd
import polars as pl
import pytest

from skcausal.datatypes.utils import (
    check_supported_types,
    collect_column_types,
    convert,
    enforce_dtypes,
    get_backend,
)


def test_get_backend_and_convert_round_trip_between_pandas_and_polars():
    pandas_df = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})

    assert get_backend(pandas_df) == "pandas"

    polars_df = convert(pandas_df, "polars")

    assert get_backend(polars_df) == "polars"
    assert polars_df.to_dict(as_series=False) == {
        "value": [1, 2],
        "label": ["a", "b"],
    }

    converted_back = convert(polars_df, "pandas")

    assert get_backend(converted_back) == "pandas"
    pd.testing.assert_frame_equal(converted_back, pandas_df)


def test_enforce_dtypes_converts_pandas_columns_with_registered_column_types():
    pandas_df = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})

    converted = enforce_dtypes(pandas_df.copy())

    assert str(converted["value"].dtype) == "float64"
    assert isinstance(converted["label"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize(
    ("dataframe", "expected_schema"),
    [
        (
            pl.DataFrame({"value": [1, 2], "label": ["a", "b"]}),
            {"value": pl.Float64, "label": pl.Categorical},
        ),
        (
            pl.DataFrame({"value": [1, 2], "label": ["a", "b"]}).with_columns(
                pl.col("label").cast(pl.Utf8)
            ),
            {"value": pl.Float64, "label": pl.Categorical},
        ),
    ],
)
def test_enforce_dtypes_converts_polars_columns_with_registered_column_types(
    dataframe, expected_schema
):
    converted = enforce_dtypes(dataframe)

    assert dict(converted.schema) == expected_schema


@pytest.mark.parametrize(
    ("df", "supported_types", "expected_messages"),
    [
        # All types supported — no messages
        (
            pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            ["continuous", "categorical"],
            [],
        ),
        # continuous not in supported_types — message for "x"
        (
            pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            ["categorical"],
            ["Column 'x' is of type 'continuous', which is not supported."],
        ),
        # categorical not in supported_types — message for "label"
        (
            pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            ["continuous"],
            ["Column 'label' is of type 'categorical', which is not supported."],
        ),
        # No types supported — messages for all columns
        (
            pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            [],
            [
                "Column 'x' is of type 'continuous', which is not supported.",
                "Column 'label' is of type 'categorical', which is not supported.",
            ],
        ),
        # Polars — all types supported — no messages
        (
            pl.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            ["continuous", "categorical"],
            [],
        ),
        # Polars — categorical not in supported_types — message for "label"
        (
            pl.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
            ["continuous"],
            ["Column 'label' is of type 'categorical', which is not supported."],
        ),
    ],
)
def test_check_supported_types_returns_messages_for_unsupported_column_types(
    df, supported_types, expected_messages
):
    messages = check_supported_types(df, supported_types, errors="return")

    assert messages == expected_messages


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
        pl.DataFrame({"x": [1.0, 2.0], "label": ["a", "b"]}),
    ],
)
def test_check_supported_types_raises_when_unsupported_columns_present(df):
    with pytest.raises(ValueError, match="Unsupported column types found"):
        check_supported_types(df, ["continuous"], errors="raise")


@pytest.mark.parametrize(
    ("dataframe", "expected_types"),
    [
        (pd.DataFrame({"flag": [True, False]}), {"flag": "categorical"}),
        (pl.DataFrame({"flag": [True, False]}), {"flag": "categorical"}),
    ],
)
def test_collect_column_types_classifies_boolean_as_categorical(
    dataframe, expected_types
):
    assert collect_column_types(dataframe) == expected_types


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
        pl.DataFrame({"when": [date(2024, 1, 1), date(2024, 1, 2)]}),
    ],
)
def test_collect_column_types_raises_when_column_type_is_unregistered(dataframe):
    with pytest.raises(ValueError, match="does not match any registered column type"):
        collect_column_types(dataframe)


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-02"])}),
        pl.DataFrame({"when": [date(2024, 1, 1), date(2024, 1, 2)]}),
    ],
)
def test_enforce_dtypes_raises_when_column_type_is_unregistered(dataframe):
    with pytest.raises(ValueError, match="does not match any registered column type"):
        enforce_dtypes(dataframe)


@pytest.mark.parametrize(
    ("dataframe", "column_types"),
    [
        (
            pd.DataFrame({"flag": [True, False], "value": [1, 2]}),
            {"flag": "categorical", "value": "continuous"},
        ),
        (
            pl.DataFrame({"flag": [True, False], "value": [1, 2]}),
            {"flag": "categorical", "value": "continuous"},
        ),
    ],
)
def test_enforce_dtypes_with_explicit_column_types_preserves_boolean_categories(
    dataframe, column_types
):
    converted = enforce_dtypes(dataframe, column_types=column_types)

    assert collect_column_types(converted) == column_types

    if get_backend(converted) == "pandas":
        assert pd.api.types.is_bool_dtype(converted["flag"])
        assert str(converted["value"].dtype) == "float64"
    else:
        assert converted.schema["flag"] == pl.Boolean
        assert converted.schema["value"] == pl.Float64


def test_enforce_dtypes_with_explicit_column_types_requires_exact_column_match():
    dataframe = pd.DataFrame({"value": [1, 2]})

    with pytest.raises(ValueError, match="column_types must match the dataframe"):
        enforce_dtypes(dataframe, column_types={"other": "continuous"})
