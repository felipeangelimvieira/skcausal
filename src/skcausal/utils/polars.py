import polars as pl


def to_dummies(df: pl.DataFrame, column: str) -> pl.DataFrame:

    for cat in df.schema[column].categories[:-1]:
        df = df.with_columns(
            pl.when(pl.col(column) == cat)
            .then(1)
            .otherwise(0)
            .cast(pl.Boolean)
            .alias(f"{column}__dummy_{cat}")
        )
    return df.drop(column)


def convert_categorical_to_dummies(df: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Enum or dtype == pl.Categorical:
            df = to_dummies(df, col)
    return df
