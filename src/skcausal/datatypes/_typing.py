from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

__all__ = ["DataFrameLike", "PandasDataFrame", "PolarsDataFrame"]


if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    DataFrameLike: TypeAlias = pd.DataFrame | pl.DataFrame
    PandasDataFrame: TypeAlias = pd.DataFrame
    PolarsDataFrame: TypeAlias = pl.DataFrame
else:
    DataFrameLike: TypeAlias = Any
    PandasDataFrame: TypeAlias = Any
    PolarsDataFrame: TypeAlias = Any
