from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Dict, Iterable, Optional

from sklearn.pipeline import Pipeline as SklearnPipeline


class Pipeline(SklearnPipeline):
    """Extension of sklearn's Pipeline that propagates sample weights."""

    def fit(
        self,
        X: Any,
        y: Optional[Iterable[Any]] = None,
        *,
        sample_weight: Optional[Iterable[Any]] = None,
        sample_weights: Optional[Iterable[Any]] = None,
        **fit_params: Any,
    ) -> "Pipeline":
        """Fit the pipeline while forwarding sample weights when supported."""
        resolved_weight = self._resolve_sample_weight(sample_weight, sample_weights)

        if resolved_weight is not None:
            fit_params = self._inject_sample_weight(fit_params, resolved_weight)

        super().fit(X, y, **fit_params)
        return self

    def _inject_sample_weight(
        self,
        fit_params: Dict[str, Any],
        sample_weight: Iterable[Any],
    ) -> Dict[str, Any]:
        updated_params: Dict[str, Any] = dict(fit_params)
        for name, step in self.steps:
            if not self._supports_sample_weight(step):
                continue

            key = f"{name}__sample_weight"
            updated_params.setdefault(key, sample_weight)

        return updated_params

    @staticmethod
    def _supports_sample_weight(step: Any) -> bool:
        if step in (None, "passthrough"):
            return False

        fit = getattr(step, "fit", None)
        if fit is None:
            return False

        try:
            sig = signature(fit)
        except (ValueError, TypeError):
            return False

        for param in sig.parameters.values():
            if param.kind == Parameter.VAR_KEYWORD:
                return True

        sample_weight_param = sig.parameters.get("sample_weight")
        if sample_weight_param is None:
            return False

        return sample_weight_param.kind in (
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        )

    @staticmethod
    def _resolve_sample_weight(
        sample_weight: Optional[Iterable[Any]],
        sample_weights: Optional[Iterable[Any]],
    ) -> Optional[Iterable[Any]]:
        if sample_weight is not None and sample_weights is not None:
            raise ValueError("Provide only one of 'sample_weight' or 'sample_weights'.")
        return sample_weight if sample_weight is not None else sample_weights
