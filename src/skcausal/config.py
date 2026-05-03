_VALID_BACKENDS = frozenset({"pandas", "polars"})

default_backend = "polars"


def get_default_backend(fallback: str = "polars") -> str:
    """Return the configured default dataframe backend.

    Parameters
    ----------
    fallback : str, default="polars"
        Backend to use when ``default_backend`` is unset.
    """

    backend = default_backend if default_backend is not None else fallback
    if not isinstance(backend, str):
        raise ValueError(
            "skcausal.config.default_backend must be a string equal to "
            "'pandas' or 'polars'."
        )

    normalized_backend = backend.lower()
    if normalized_backend not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(
            "skcausal.config.default_backend must be one of "
            f"{{{valid}}}. Got {backend!r}."
        )

    return normalized_backend
