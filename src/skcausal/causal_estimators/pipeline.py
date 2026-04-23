from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from skbase.base._meta import _MetaObjectMixin

import numpy as np

from skcausal.causal_estimators.base import BaseAverageCausalResponseEstimator
from skcausal.transformations.base import BaseTransformation

__all__ = ["Pipeline"]


class _IdentityTransformation(BaseTransformation):
    """Private identity transformation used in automated Pipeline tests."""

    def _fit(self, X):
        return self

    def _transform(self, X):
        return X


class _MeanResponseEstimator(BaseAverageCausalResponseEstimator):
    """Private mean-response estimator used in automated Pipeline tests."""

    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": True,
    }

    def _fit(self, X, t, y):
        self.mean_response_ = float(
            np.asarray(y.to_numpy(), dtype=float).reshape(-1).mean()
        )
        return self

    def _predict(self, X, t):
        return np.full(len(t), self.mean_response_, dtype=float)


class Pipeline(_MetaObjectMixin, BaseAverageCausalResponseEstimator):
    """Compose preprocessing transformations with a final causal-response estimator.

    Parameters
    ----------
    steps : sequence of tuples
            Pipeline specification. Each transformation step must be provided as
            ``(name, transformation, apply_to)`` where ``transformation`` is a
            ``BaseTransformation`` and ``apply_to`` is either ``"X"``, ``"t"``,
            or ``"y"``. The last step must be a
            ``BaseAverageCausalResponseEstimator`` provided as either
            ``(name, estimator)`` or ``(name, estimator, None)``.
    """

    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": False,
    }

    def __init__(self, steps: Sequence[tuple[Any, ...]]):
        self.steps = steps

        super().__init__()

        self.set_tags(
            named_object_parameters="steps",
            fitted_named_object_parameters="steps_",
        )

        (
            self._transform_step_specs,
            self._final_step_name,
            self._final_estimator,
        ) = self._validate_steps(steps)
        self._set_pipeline_tags()

    def _set_pipeline_tags(self) -> None:
        pipeline_tags = self.get_tags()
        estimator_tags = self._final_estimator.get_tags()

        self.set_tags(
            backend=estimator_tags.get("backend", pipeline_tags.get("backend")),
            **{
                "capability:t_type": estimator_tags.get(
                    "capability:t_type",
                    pipeline_tags.get("capability:t_type"),
                ),
                "capability:multidimensional_treatment": estimator_tags.get(
                    "capability:multidimensional_treatment",
                    pipeline_tags.get("capability:multidimensional_treatment"),
                ),
                "one_hot_encode_enum_columns": estimator_tags.get(
                    "one_hot_encode_enum_columns",
                    pipeline_tags.get("one_hot_encode_enum_columns", False),
                ),
            },
        )

    def fit(self, X, t, y):
        """Fit all transformations and the final causal-response estimator."""
        X_transformed, t_transformed, y_transformed = X, t, y

        self.transform_steps_ = []
        self.named_steps_ = {}

        for name, transformation, apply_to in self._transform_step_specs:
            fitted_transformation = transformation.clone()

            if apply_to == "X":
                fitted_transformation.fit(X_transformed)
                X_transformed = fitted_transformation.transform(X_transformed)
            elif apply_to == "t":
                fitted_transformation.fit(t_transformed)
                t_transformed = fitted_transformation.transform(t_transformed)
            else:
                fitted_transformation.fit(y_transformed)
                y_transformed = fitted_transformation.transform(y_transformed)

            self.transform_steps_.append((name, fitted_transformation, apply_to))
            self.named_steps_[name] = fitted_transformation

        self.estimator_ = self._final_estimator.clone()
        self.estimator_.fit(X_transformed, t_transformed, y_transformed)
        self.named_steps_[self._final_step_name] = self.estimator_
        self.steps_ = [
            *self.transform_steps_,
            (self._final_step_name, self.estimator_),
        ]
        return self

    def predict(self, X, t):
        """Apply fitted transformations and delegate response prediction."""
        X_transformed, t_transformed = self._transform_inputs(X, t)
        return np.asarray(self.estimator_.predict(X_transformed, t_transformed))

    def _fit(self, X, t, y):
        raise NotImplementedError(
            "Pipeline overrides fit directly and does not use _fit."
        )

    def _predict(self, X, t):
        raise NotImplementedError(
            "Pipeline overrides predict directly and does not use _predict."
        )

    def _set_params(self, attr: str, **params):
        """Delegate to parent _set_params, preserving apply_to in 3-tuple steps."""
        if attr in params:
            new_steps = params[attr]
            params[attr] = [(s[0], s[1]) + s[2:] for s in new_steps]
        return super()._set_params(attr, **params)

    def _transform_inputs(self, X, t):
        X_transformed, t_transformed = X, t

        for _, transformation, apply_to in self.transform_steps_:
            if apply_to == "X":
                X_transformed = transformation.transform(X_transformed)
            elif apply_to == "t":
                t_transformed = transformation.transform(t_transformed)

        return X_transformed, t_transformed

    def _dunder_concat(
        self,
        other,
        base_class=None,
        composite_class=None,
        attr_name="steps",
        concat_order="left",
        composite_params=None,
    ):
        """Concatenate transformation step specs while preserving apply_to metadata."""
        if attr_name != "steps":
            return super()._dunder_concat(
                other,
                base_class=base_class,
                composite_class=composite_class,
                attr_name=attr_name,
                concat_order=concat_order,
                composite_params=composite_params,
            )

        if concat_order not in ["left", "right"]:
            raise ValueError(
                f"`concat_order` must be 'left' or 'right', but found {concat_order!r}."
            )

        self_transform_steps, self_density_step = self._split_steps_for_concat(
            self.steps,
            allow_missing_density=False,
        )
        other_transform_steps, other_density_step = self._coerce_concat_other(other)

        if other_transform_steps is NotImplemented:
            return NotImplemented

        if self_density_step is not None and other_density_step is not None:
            raise ValueError(
                "Pipeline concatenation supports exactly one average-response estimator across operands."
            )

        final_density_step = self_density_step
        if final_density_step is None:
            final_density_step = other_density_step

        if final_density_step is None:
            raise ValueError(
                "Pipeline concatenation requires a final BaseAverageCausalResponseEstimator step."
            )

        if concat_order == "left":
            transform_steps = [*self_transform_steps, *other_transform_steps]
        else:
            transform_steps = [*other_transform_steps, *self_transform_steps]

        if composite_params is None:
            composite_params = {}
        else:
            composite_params = composite_params.copy()

        composite_params.update({"steps": [*transform_steps, final_density_step]})
        return self.__class__(**composite_params)

    def _coerce_concat_other(self, other):
        if isinstance(other, self.__class__):
            return self._split_steps_for_concat(
                other.steps, allow_missing_density=False
            )

        if isinstance(other, tuple):
            return self._split_steps_for_concat([other], allow_missing_density=True)

        if isinstance(other, BaseTransformation):
            raise TypeError(
                "Concatenating a BaseTransformation requires a "
                "(name, transformation, apply_to) step tuple."
            )

        if isinstance(other, BaseAverageCausalResponseEstimator):
            return [], (type(other).__name__, other)

        return NotImplemented, None

    def _split_steps_for_concat(self, steps, allow_missing_density):
        transform_steps = []
        density_step = None

        for index, step in enumerate(steps):
            if not isinstance(step, tuple):
                raise TypeError("Each step must be provided as a tuple.")

            if len(step) not in {2, 3}:
                raise ValueError(
                    "Each step must be either (name, estimator) or "
                    "(name, transformation, apply_to)."
                )

            name = step[0]
            estimator = step[1]

            if isinstance(estimator, BaseAverageCausalResponseEstimator):
                if index != len(steps) - 1:
                    raise ValueError(
                        "A BaseAverageCausalResponseEstimator step must be the final pipeline step."
                    )
                if len(step) == 3 and step[2] is not None:
                    raise ValueError(
                        "The final BaseAverageCausalResponseEstimator step does not accept apply_to."
                    )
                density_step = (name, estimator)
                continue

            if not isinstance(estimator, BaseTransformation):
                raise TypeError("Non-final steps must be BaseTransformation instances.")

            if len(step) != 3 or step[2] not in {"X", "t", "y"}:
                raise ValueError("apply_to must be either 'X', 't', or 'y'.")

            transform_steps.append((name, estimator, step[2]))

        if density_step is None and not allow_missing_density:
            raise TypeError(
                "The last pipeline step must be a BaseAverageCausalResponseEstimator instance."
            )

        return transform_steps, density_step

    def _validate_steps(self, steps: Sequence[tuple[Any, ...]]):
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise TypeError("steps must be a non-string sequence of step tuples.")

        if len(steps) == 0:
            raise ValueError("steps must contain at least one step.")

        transform_step_specs = []
        final_step_name = None
        final_estimator = None
        step_names = []

        for index, step in enumerate(steps):
            if not isinstance(step, tuple):
                raise TypeError("Each step must be provided as a tuple.")
            if len(step) not in {2, 3}:
                raise ValueError(
                    "Each step must be either (name, estimator) or "
                    "(name, transformation, apply_to)."
                )

            name = step[0]
            estimator = step[1]

            if not isinstance(name, str):
                raise TypeError("Step names must be strings.")
            step_names.append(name)

            if isinstance(estimator, BaseAverageCausalResponseEstimator):
                if index != len(steps) - 1:
                    raise ValueError(
                        "A BaseAverageCausalResponseEstimator step must be the final pipeline step."
                    )
                if len(step) == 3 and step[2] is not None:
                    raise ValueError(
                        "The final BaseAverageCausalResponseEstimator step does not accept apply_to."
                    )
                final_step_name = name
                final_estimator = estimator
                continue

            if not isinstance(estimator, BaseTransformation):
                raise TypeError("Non-final steps must be BaseTransformation instances.")

            if len(step) != 3:
                raise ValueError(
                    "Transformation steps must be provided as "
                    "(name, transformation, apply_to)."
                )

            apply_to = step[2]
            if apply_to not in {"X", "t", "y"}:
                raise ValueError("apply_to must be either 'X', 't', or 'y'.")

            transform_step_specs.append((name, estimator, apply_to))

        self._check_names(step_names, make_unique=False)

        if final_estimator is None:
            raise TypeError(
                "The last pipeline step must be a BaseAverageCausalResponseEstimator instance."
            )

        return transform_step_specs, final_step_name, final_estimator

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {
                "steps": [
                    ("transform_X", _IdentityTransformation(), "X"),
                    ("estimator", _MeanResponseEstimator()),
                ]
            },
            {
                "steps": [
                    ("transform_y", _IdentityTransformation(), "y"),
                    ("estimator", _MeanResponseEstimator()),
                ]
            },
        ]
