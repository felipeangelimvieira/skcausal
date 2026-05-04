from skcausal.datatypes._backend import _CONVERSION_FUNCTIONS, BaseBackend

__all__ = [
    "get_backend",
    "enforce_dtypes",
    "convert",
    "check_supported_types",
    "collect_column_types",
]


def get_backend(df):
    """
    Get the backend of a dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe to get the backend of. Can be any dataframe supported by
        the registered backends.

    Returns
    -------
    str
        The name of the backend.

    """
    for backend_name, backend_class in BaseBackend.backends_dict().items():
        if backend_class().isinstance(df):
            return backend_name

    raise ValueError(f"Unsupported dataframe type {type(df)}.")


def enforce_dtypes(df, column_types: dict[str, str] | None = None):
    """
    Enforce registered column types on the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe to enforce the column types on.
    column_types : dict, optional
        A dictionary where keys are column names and values are the desired
        column types (e.g., "continuous", "categorical"). When omitted,
        column types are inferred from the existing dataframe dtypes and then
        normalized to the backend's canonical representation.

    Returns
    -------
    DataFrame
        The dataframe with enforced column types.

    """

    backend_name = get_backend(df)
    backend = BaseBackend.backends_dict()[backend_name]()

    if column_types is None:
        return backend.convert_column_types(df)

    normalized_column_types = dict(column_types)
    df_columns = list(df.columns)
    if set(df_columns) != set(normalized_column_types):
        missing = [
            column for column in df_columns if column not in normalized_column_types
        ]
        extra = [
            column for column in normalized_column_types if column not in df_columns
        ]
        problems = []
        if missing:
            problems.append(f"missing column types for {missing}")
        if extra:
            problems.append(f"unexpected column types for {extra}")
        raise ValueError(
            "column_types must match the dataframe columns exactly: "
            + "; ".join(problems)
            + "."
        )

    backend_column_types = backend.get_column_types()
    for column in df_columns:
        column_type_name = normalized_column_types[column]
        if column_type_name not in backend_column_types:
            valid = ", ".join(sorted(backend_column_types))
            raise ValueError(
                f"Unsupported column type {column_type_name!r} for backend "
                f"{backend_name!r}. Expected one of {{{valid}}}."
            )
        df = backend_column_types[column_type_name].convert_dataframe_column(df, column)

    return df


def convert(df, backend: str):
    """
    Convert a dataframe to the specified backend.

    Parameters
    ----------
    df : DataFrame
        The dataframe to convert. Can be any dataframe supported by
        the registered backends.
    backend : str
        The target backend. Options are "pandas" and "polars".

    Returns
    -------
    DataFrame
        The converted dataframe.

    """
    if isinstance(backend, str):
        backend = backend.lower()
    else:
        raise ValueError("Backend must be a string.")

    current_backend = get_backend(df)

    if current_backend == backend:
        return df

    conversion_func = _CONVERSION_FUNCTIONS.get((current_backend, backend))
    if conversion_func is None:
        raise ValueError(
            f"Conversion from {current_backend} to {backend} is not supported."
        )

    return conversion_func(df)


def check_supported_types(df, supported_types, errors="raise"):
    """
    Check if the dataframe has columns of supported types.

    Collect messages in a list and raise error if errors is "raise",
    otherwise return the list of messages.

    Parameters
    ----------
    df : DataFrame
        The dataframe to check.
    supported_types : list
        A list of supported types (e.g., ["continuous", "categorical"]).
    errors : str, optional
        Whether to raise an error if unsupported types are found. Options are
        "raise" and "return". Default is "raise".
    """

    column_types = collect_column_types(df)

    messages = []
    for column in df.columns:
        if column_types[column] not in supported_types:
            messages.append(
                f"Column '{column}' is of type '{column_types[column]}', which is not supported."
            )
    if errors == "raise" and messages:
        raise ValueError("Unsupported column types found:\n" + "\n".join(messages))
    return messages


def collect_column_types(df):
    """
    Collect the column types of a dataframe.

    Returns a dict of {"column_name" : type}

    Parameters
    ----------
    df : DataFrame
        The dataframe to collect column types from.

    Returns
    -------
    dict
        A dictionary where keys are column names and values are the detected
        column types (e.g., "continuous", "categorical").
    """

    backend_name = get_backend(df)
    backend = BaseBackend.backends_dict()[backend_name]()
    column_type_classes = backend.get_column_types()

    types = {}
    for column in df.columns:
        matched_column_type = None
        for col_type_name, col_type_class in column_type_classes.items():
            if col_type_class.is_column_instanceof(df, column):
                matched_column_type = col_type_name
                break

        if matched_column_type is None:
            raise ValueError(
                f"Column '{column}' with dtype {df[column].dtype!r} does not match "
                f"any registered column type for backend '{backend_name}'."
            )

        types[column] = matched_column_type

    return types
