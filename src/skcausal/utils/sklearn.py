from sklearn.pipeline import Pipeline


def _resolve_sample_weight_fit_args(estimator, sample_weight) -> dict:
    """Return kwargs to pass sample_weight to estimator.fit (or nested)."""
    if sample_weight is None:
        return {}

    # direct support
    try:
        sig = estimator.fit.__signature__
    except AttributeError:
        from inspect import signature

        try:
            sig = signature(estimator.fit)
        except (TypeError, ValueError):
            sig = None

    if sig is not None and "sample_weight" in sig.parameters:
        return {"sample_weight": sample_weight}

    # Pipelines
    if isinstance(estimator, Pipeline):
        for name, step in estimator.steps:
            sub = _resolve_sample_weight_fit_args(step, sample_weight)
            if sub:
                # route into that step
                return {f"{name}__{k}": v for k, v in sub.items()}

    # Common nested patterns
    for attr in ("estimator", "base_estimator"):
        nested = getattr(estimator, attr, None)
        if nested is not None:
            sub = _resolve_sample_weight_fit_args(nested, sample_weight)
            if sub:
                return sub

    return {}
