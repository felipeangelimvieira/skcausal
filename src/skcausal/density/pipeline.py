from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from skbase.base._meta import _MetaObjectMixin

from skcausal.density.base import BaseDensityEstimator
from skcausal.transformations.base import BaseTransformation

__all__ = ["Pipeline"]


class _IdentityTransformation(BaseTransformation):
    """Private identity transformation used in automated Pipeline tests."""

    def _fit(self, X):
        return self

    def _transform(self, X):
        return X


class Pipeline(_MetaObjectMixin, BaseDensityEstimator):
    """Compose preprocessing transformations with a final density estimator.

    Parameters
    ----------
    steps : sequence of tuples
            Pipeline specification. Each transformation step must be provided as
            ``(name, transformation, apply_to)`` where ``transformation`` is a
            ``BaseTransformation`` and ``apply_to`` is either ``"X"`` or ``"t"``.
            The last step must be a ``BaseDensityEstimator`` provided as either
            ``(name, estimator)`` or ``(name, estimator, None)``.
    """

    _tags = {
        "backend": "polars",
        "capability:t_type": ["continuous", "categorical"],
        "capability:multidimensional_treatment": True,
        "density_kind": "conditional",
        "named_object_parameters": "steps",
        "fitted_named_object_parameters": "steps_",
    }

    def __init__(self, steps: Sequence[tuple[Any, ...]]):
        self.steps = steps

        super().__init__()

        (
            self._transform_step_specs,
            self._final_step_name,
            self._final_density_estimator,
        ) = self._validate_steps(steps)
        self._set_pipeline_tags()

    def _set_pipeline_tags(self) -> None:
        self.set_tags(
            backend=self._final_density_estimator.get_tag(
                "backend",
                self.get_tag("backend"),
            ),
            **{
                "capability:t_type": self._final_density_estimator.get_tag(
                    "capability:t_type",
                    self.get_tag("capability:t_type"),
                ),
                "capability:multidimensional_treatment": self._final_density_estimator.get_tag(
                    "capability:multidimensional_treatment",
                    self.get_tag("capability:multidimensional_treatment"),
                ),
            },
            density_kind=self._final_density_estimator.get_tag(
                "density_kind",
                self.get_tag("density_kind"),
            ),
        )

    def fit(self, X, t):
        """Fit all transformations and the final density estimator."""
        X_transformed, t_transformed = X, t

        self.transform_steps_ = []
        self.named_steps_ = {}

        for name, transformation, apply_to in self._transform_step_specs:
            fitted_transformation = transformation.clone()

            if apply_to == "X":
                fitted_transformation.fit(X_transformed)
                X_transformed = fitted_transformation.transform(X_transformed)
            else:
                fitted_transformation.fit(t_transformed)
                t_transformed = fitted_transformation.transform(t_transformed)

            self.transform_steps_.append((name, fitted_transformation, apply_to))
            self.named_steps_[name] = fitted_transformation

        self.density_estimator_ = self._final_density_estimator.clone()
        self.density_estimator_.fit(X_transformed, t_transformed)
        self.named_steps_[self._final_step_name] = self.density_estimator_
        self.steps_ = [
            *self.transform_steps_,
            (self._final_step_name, self.density_estimator_),
        ]
        return self

    def predict_density(self, X, t):
        """Apply fitted transformations and delegate density prediction."""
        X_transformed, t_transformed = self._transform_inputs(X, t)
        return self.density_estimator_.predict_density(X_transformed, t_transformed)

    def _fit(self, X, t):
        raise NotImplementedError(
            "Pipeline overrides fit directly and does not use _fit."
        )

    def _predict_density(self, X, t):
        raise NotImplementedError(
            "Pipeline overrides predict_density directly and does not use _predict_density."
        )

    def _set_params(self, attr: str, **params):
        """Set params while treating step tuples as ``(name, obj, ...)``."""
        if not params:
            return self

        # Mirror skbase's reset semantics when we consume replacement params here.
        params_handled_locally = False

        if attr in params:
            setattr(self, attr, params.pop(attr))
            params_handled_locally = True

        items = getattr(self, attr, None)
        names = []
        if isinstance(items, dict):
            names = list(items.keys())
        elif items and isinstance(items, (list, tuple)):
            # Pipelines store steps as (name, obj, apply_to), so only read index 0.
            names = [item[0] for item in items]

        for name in list(params.keys()):
            if "__" not in name and name in names:
                self._replace_object(attr, name, params.pop(name))
                params_handled_locally = True

        # Skip _MetaObjectMixin.set_params because it assumes named objects are 2-tuples.
        super(_MetaObjectMixin, self).set_params(**params)

        if params_handled_locally and not params:
            self.reset()

        return self

    def _get_params(self, attr, deep=True, fitted=False):
        """Get params while treating step tuples as ``(name, obj, ...)``."""
        if fitted:
            method_shallow = "_get_fitted_params"
            method_public = "get_fitted_params"
            deepkw = {}
        else:
            method_shallow = "get_params"
            method_public = "get_params"
            deepkw = {"deep": deep}

        # Start from BaseObject/BaseEstimator params, bypassing skbase's 2-tuple coercion.
        out = getattr(super(_MetaObjectMixin, self), method_shallow)(**deepkw)

        if deep and hasattr(self, attr):
            named_objects = getattr(self, attr)
            if isinstance(named_objects, dict):
                named_objects_ = list(named_objects.items())
            else:
                # Expose only (name, obj) to the parameter API while keeping apply_to internal.
                named_objects_ = [
                    (obj[0], obj[1])
                    for obj in named_objects
                    if isinstance(obj, tuple) and len(obj) >= 2
                ]

            out.update(named_objects_)
            for name, obj in named_objects_:
                cond1 = hasattr(obj, method_public)
                is_fitted = hasattr(obj, "is_fitted") and obj.is_fitted
                cond2 = not fitted or is_fitted
                if cond1 and cond2:
                    for key, value in getattr(obj, method_public)(**deepkw).items():
                        out[f"{name}__{key}"] = value

        return out

    def _transform_inputs(self, X, t):
        X_transformed, t_transformed = X, t

        for _, transformation, apply_to in self.transform_steps_:
            if apply_to == "X":
                X_transformed = transformation.transform(X_transformed)
            else:
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
                "Pipeline concatenation supports exactly one density estimator across operands."
            )

        final_density_step = self_density_step
        if final_density_step is None:
            final_density_step = other_density_step

        if final_density_step is None:
            raise ValueError(
                "Pipeline concatenation requires a final BaseDensityEstimator step."
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

        if isinstance(other, BaseDensityEstimator):
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

            if isinstance(estimator, BaseDensityEstimator):
                if index != len(steps) - 1:
                    raise ValueError(
                        "A BaseDensityEstimator step must be the final pipeline step."
                    )
                density_step = (name, estimator)
                continue

            if not isinstance(estimator, BaseTransformation):
                raise TypeError("Non-final steps must be BaseTransformation instances.")

            if len(step) != 3 or step[2] not in {"X", "t"}:
                raise ValueError("apply_to must be either 'X' or 't'.")

            transform_steps.append((name, estimator, step[2]))

        if density_step is None and not allow_missing_density:
            raise TypeError(
                "The last pipeline step must be a BaseDensityEstimator instance."
            )

        return transform_steps, density_step

    def _validate_steps(self, steps: Sequence[tuple[Any, ...]]):
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise TypeError("steps must be a non-string sequence of step tuples.")

        if len(steps) == 0:
            raise ValueError("steps must contain at least one step.")

        transform_step_specs = []
        final_step_name = None
        final_density_estimator = None
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

            if isinstance(estimator, BaseDensityEstimator):
                if index != len(steps) - 1:
                    raise ValueError(
                        "A BaseDensityEstimator step must be the final pipeline step."
                    )
                if len(step) == 3 and step[2] is not None:
                    raise ValueError(
                        "The final BaseDensityEstimator step does not accept apply_to."
                    )
                final_step_name = name
                final_density_estimator = estimator
                continue

            if not isinstance(estimator, BaseTransformation):
                raise TypeError("Non-final steps must be BaseTransformation instances.")

            if len(step) != 3:
                raise ValueError(
                    "Transformation steps must be provided as "
                    "(name, transformation, apply_to)."
                )

            apply_to = step[2]
            if apply_to not in {"X", "t"}:
                raise ValueError("apply_to must be either 'X' or 't'.")

            transform_step_specs.append((name, estimator, apply_to))

        self._check_names(step_names, make_unique=False)

        if final_density_estimator is None:
            raise TypeError(
                "The last pipeline step must be a BaseDensityEstimator instance."
            )

        return transform_step_specs, final_step_name, final_density_estimator

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skcausal.density.naive import NaiveDensityEstimator

        return [
            {
                "steps": [
                    ("transform_X", _IdentityTransformation(), "X"),
                    ("density", NaiveDensityEstimator()),
                ]
            },
            {
                "steps": [
                    ("transform_t", _IdentityTransformation(), "t"),
                    ("density", NaiveDensityEstimator(density_kind="stabilized")),
                ]
            },
        ]
