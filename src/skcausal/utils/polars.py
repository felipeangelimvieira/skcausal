import polars as pl

INTEGER_DTYPES = [
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
]

FLOAT_DTYPES = [pl.Float32, pl.Float64]

ALL_DTYPES = [pl.Boolean, pl.Enum, *INTEGER_DTYPES, *FLOAT_DTYPES]


def assert_schema_equal(df_a: pl.Schema, df_b: pl.Schema) -> bool:
    assert df_a == df_b


def to_dummies(df: pl.DataFrame, column: str) -> pl.DataFrame:

    for cat in df.schema[column].categories[:-1]:
        df = df.with_columns(
            pl.when(pl.col(column) == cat)
            .then(1)
            .otherwise(0)
            .cast(pl.Binary)
            .alias(f"{column}__dummy_{cat}")
        )
    return df.drop(column)


def convert_categorical_to_dummies(df: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Enum or dtype == pl.Categorical:
            df = to_dummies(df, col)
    return df
