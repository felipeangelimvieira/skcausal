from skbase.base import BaseObject

__all__ = [
    "BaseColumnType",
    "PandasContinuousColumnType",
    "PandasCategoricalColumnType",
    "PolarsContinuousColumnType",
    "PolarsCategoricalColumnType",
]


class BaseColumnType(BaseObject):
    _tags = {"object_type": "column_type", "name": None, "backend": None}

    @classmethod
    def is_column_instanceof(cls, df, col): ...


class PandasColumnTypeMixin:

    @classmethod
    def convert_dataframe_column(cls, df, col):
        """Convert the specified column in the dataframe to the appropriate dtype"""
        df = df.copy()
        df[col] = cls.convert_column(df[col])
        return df

    @classmethod
    def is_column_instanceof(cls, df, col):
        """Check if the specified column in the dataframe is an instance of the column type"""
        return cls.isinstance(df[col])


class PolarsColumnTypeMixin:

    @classmethod
    def convert_dataframe_column(cls, df, col):
        """Convert the specified column in the dataframe to the appropriate dtype"""
        return df.with_columns(cls.convert_column(df[col]).alias(col))

    @classmethod
    def is_column_instanceof(cls, df, col):
        """Check if the specified column in the dataframe is an instance of the column type"""
        return cls.isinstance(df[col])


class PandasContinuousColumnType(PandasColumnTypeMixin, BaseColumnType):
    _tags = {
        "object_type": "column_type",
        "name": "continuous",
        "backend": "pandas",
    }

    @classmethod
    def isinstance(cls, obj):
        import pandas as pd

        return isinstance(obj, pd.Series) and (
            pd.api.types.is_numeric_dtype(obj) and not pd.api.types.is_bool_dtype(obj)
        )

    @classmethod
    def column_mask(cls, df):
        import pandas as pd

        return df.select_dtypes(include=["number"]).columns

    @classmethod
    def convert_column(cls, col):
        """Convert to float64"""
        return col.astype("float64")


class PandasCategoricalColumnType(PandasColumnTypeMixin, BaseColumnType):
    _tags = {
        "object_type": "column_type",
        "name": "categorical",
        "backend": "pandas",
    }

    @classmethod
    def isinstance(cls, obj):
        import pandas as pd

        return isinstance(obj, pd.Series) and (
            pd.api.types.is_categorical_dtype(obj)
            or pd.api.types.is_bool_dtype(obj)
            or pd.api.types.is_object_dtype(obj)
            or pd.api.types.is_string_dtype(obj)
        )

    @classmethod
    def column_mask(cls, df):
        import pandas as pd

        return df.select_dtypes(include=["bool", "category", "object"]).columns

    @classmethod
    def convert_column(cls, col):
        """Convert to category dtype"""
        import pandas as pd

        if pd.api.types.is_bool_dtype(col):
            return col.astype(bool)
        return col.astype("category")


class PolarsContinuousColumnType(PolarsColumnTypeMixin, BaseColumnType):
    _tags = {
        "object_type": "column_type",
        "name": "continuous",
        "backend": "polars",
    }

    @classmethod
    def isinstance(cls, obj):
        import polars as pl

        dtype = obj.dtype

        return isinstance(obj, pl.Series) and dtype in [
            pl.Float16,
            pl.Float32,
            pl.Float64,
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]

    @classmethod
    def column_mask(cls, df):
        import polars as pl

        return [col for col, dtype in df.schema.items() if cls.isinstance(df[col])]

    @classmethod
    def convert_column(cls, col):
        """Convert to Float64"""

        import polars as pl

        return col.cast(pl.Float64)


class PolarsCategoricalColumnType(PolarsColumnTypeMixin, BaseColumnType):
    _tags = {
        "object_type": "column_type",
        "name": "categorical",
        "backend": "polars",
    }

    @classmethod
    def isinstance(cls, obj):
        import polars as pl

        dtype = obj.dtype

        return isinstance(obj, pl.Series) and dtype in [
            pl.Categorical,
            pl.Enum,
            pl.String,
            pl.Utf8,
            pl.Boolean,
        ]

    @classmethod
    def column_mask(cls, df):
        import polars as pl

        return [col for col, _ in df.schema.items() if cls.isinstance(df[col])]

    @classmethod
    def convert_column(cls, col):
        """Convert to Categorical"""

        import polars as pl

        if col.dtype == pl.Boolean:
            return col.cast(pl.Boolean)
        return col.cast(pl.Categorical)
