from skbase.base import BaseObject

__all__ = ["BaseBackend", "PandasBackend", "PolarsBackend"]


class BaseBackend(BaseObject):

    _tags = {"object_type": "backend", "name": None}

    @staticmethod
    def backends_dict():
        """
        Iterates over subclasses of BaseBackend and returns a dict.

        Returns
        -------
        dict
            A dictionary where keys are backend names and values are the
            corresponding backend classes.
        """
        backends = {}
        for subclass in BaseBackend.__subclasses__():
            name = subclass.get_class_tag("name")
            backends[name] = subclass
        return backends

    @classmethod
    def get_column_types(cls):
        """
        Get the column types for a specific backend.

        Parameters
        ----------
        backend : str
            The name of the backend to get the column types for.

        Returns
        -------
        dict
            A dictionary where keys are column type names and values are the
            corresponding column type classes that are compatible with the specified
            backend.
        """
        from skcausal.datatypes._column_type import BaseColumnType

        column_types = {}
        for subclass in BaseColumnType.__subclasses__():
            if subclass.get_class_tag("backend") == cls.get_class_tag("name"):
                column_types[subclass.get_class_tag("name")] = subclass
        return column_types

    @classmethod
    def convert_column_types(self, df):

        column_type_classes = self.get_column_types()

        for df_col in df.columns:
            matched_col_type = None
            for col_type in column_type_classes.values():
                if col_type.is_column_instanceof(df, df_col):
                    matched_col_type = col_type
                    break

            if matched_col_type is None:
                raise ValueError(
                    f"Column {df_col} of type {df[df_col].dtype} does not match any registered column type for backend {self.get_class_tag('name')}."
                )

            df = matched_col_type.convert_dataframe_column(df, df_col)

        return df


class PandasBackend(BaseBackend):

    _tags = {"object_type": "backend", "name": "pandas"}

    def isinstance(self, obj):
        import pandas as pd

        return isinstance(obj, pd.DataFrame)


class PolarsBackend(BaseBackend):
    _tags = {"object_type": "backend", "name": "polars"}

    def isinstance(self, obj):
        import polars as pl

        return isinstance(obj, pl.DataFrame)


def _from_polars_to_pandas(df):
    return df.to_pandas()


def _from_pandas_to_polars(df):
    import polars as pl

    return pl.from_pandas(df)


_CONVERSION_FUNCTIONS = {
    ("polars", "pandas"): _from_polars_to_pandas,
    ("pandas", "polars"): _from_pandas_to_polars,
}
