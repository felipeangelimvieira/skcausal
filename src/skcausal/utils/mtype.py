from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import polars as pl


__all__ = [
    "MtypeConversionError",
    "BaseConversionPolicy",
    "NumpyArrayToPolarsPolicy",
    "PolarsDataFrameToNumpyPolicy",
    "MtypeConverter",
    "default_converter",
    "convert_mtype",
]


class MtypeConversionError(RuntimeError):
    """Raised when an mtype conversion cannot be performed."""


class BaseConversionPolicy(ABC):
    """Abstract base class for conversion policies between inner mtypes."""

    from_type: Type[Any]
    to_type: Type[Any]
    name: str

    def __init__(
        self, from_type: Type[Any], to_type: Type[Any], *, name: Optional[str] = None
    ) -> None:
        self.from_type = from_type
        self.to_type = to_type
        self.name = name or f"{from_type.__name__}->{to_type.__name__}"

    def __call__(self, value: Any, **kwargs: Any) -> Any:
        self._validate_input(value)
        converted = self._convert(value, **kwargs)
        self._validate_output(converted)
        return converted

    def _validate_input(self, value: Any) -> None:
        if not isinstance(value, self.from_type):
            raise TypeError(
                f"Conversion policy '{self.name}' expects value of type {self.from_type.__name__}, "
                f"got {type(value).__name__}."
            )

    def _validate_output(self, value: Any) -> None:
        if not isinstance(value, self.to_type):
            raise TypeError(
                f"Conversion policy '{self.name}' produced value of type {type(value).__name__}, "
                f"expected {self.to_type.__name__}."
            )

    @abstractmethod
    def _convert(self, value: Any, **kwargs: Any) -> Any:
        """Concrete conversion logic implemented by subclasses."""


class NumpyArrayToPolarsPolicy(BaseConversionPolicy):
    """Conversion policy from ``numpy.ndarray`` to ``polars.DataFrame``."""

    def __init__(self) -> None:
        super().__init__(np.ndarray, pl.DataFrame, name="numpy_array->polars_dataframe")

    def _convert(
        self,
        value: np.ndarray,
        *,
        column_names: Optional[Sequence[str]] = None,
        column_prefix: str = "col",
    ) -> pl.DataFrame:
        array = np.asarray(value)

        if array.ndim > 2:
            raise ValueError(
                "Only 1D or 2D numpy arrays can be converted to polars.DataFrame; "
                f"received array with ndim={array.ndim}."
            )

        if array.ndim == 1:
            if column_names is not None and len(column_names) != 1:
                raise ValueError(
                    "For 1D arrays, column_names must be a single-element sequence if provided."
                )
            column_name = column_names[0] if column_names else "value"
            return pl.DataFrame({column_name: array})

        # array.ndim == 2 from here on
        n_columns = array.shape[1]
        if column_names is not None:
            if len(column_names) != n_columns:
                raise ValueError(
                    f"column_names length ({len(column_names)}) does not match array width ({n_columns})."
                )
            names = list(column_names)
        else:
            names = [f"{column_prefix}_{idx}" for idx in range(n_columns)]

        data = {name: array[:, idx] for idx, name in enumerate(names)}
        return pl.DataFrame(data)


class PolarsDataFrameToNumpyPolicy(BaseConversionPolicy):
    """Conversion policy from ``polars.DataFrame`` to ``numpy.ndarray``."""

    def __init__(self) -> None:
        super().__init__(pl.DataFrame, np.ndarray, name="polars_dataframe->numpy_array")

    def _convert(
        self,
        value: pl.DataFrame,
        *,
        copy: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> np.ndarray:
        numpy_array = value.to_numpy()

        if copy:
            array = np.array(numpy_array, copy=True)
        else:
            array = np.asarray(numpy_array)

        if dtype is not None:
            array = array.astype(dtype, copy=False)

        return array


ConversionKey = Tuple[Type[Any], Type[Any]]


class MtypeConverter:
    """Registry and dispatcher for converting between inner mtypes."""

    def __init__(
        self, *, policies: Optional[Iterable[BaseConversionPolicy]] = None
    ) -> None:
        self._policies: Dict[ConversionKey, BaseConversionPolicy] = {}
        if policies is not None:
            for policy in policies:
                self.register(policy)

    @staticmethod
    def _normalize_target_type(
        to_type: Union[Type[Any], Tuple[Type[Any], ...]],
    ) -> Tuple[Type[Any], ...]:
        if isinstance(to_type, tuple):
            return to_type
        return (to_type,)

    def register(
        self, policy: BaseConversionPolicy, *, overwrite: bool = False
    ) -> None:
        key = (policy.from_type, policy.to_type)
        if not overwrite and key in self._policies:
            raise ValueError(
                f"A conversion policy for {policy.from_type.__name__}->{policy.to_type.__name__} is already registered."
            )
        self._policies[key] = policy

    def unregister(self, from_type: Type[Any], to_type: Type[Any]) -> None:
        self._policies.pop((from_type, to_type), None)

    def available_conversions(self) -> Dict[ConversionKey, str]:
        return {
            (from_t, to_t): policy.name
            for (from_t, to_t), policy in self._policies.items()
        }

    def convert(
        self,
        value: Any,
        to_type: Union[Type[Any], Tuple[Type[Any], ...]],
        **kwargs: Any,
    ) -> Any:
        target_types = self._normalize_target_type(to_type)
        if any(isinstance(value, target) for target in target_types):
            return value

        start_type = type(value)
        policy_path = self._find_conversion_path(start_type, target_types)

        if policy_path is None:
            pretty_targets = ", ".join(t.__name__ for t in target_types)
            raise MtypeConversionError(
                f"No conversion path registered from {start_type.__name__} to {pretty_targets}."
            )

        converted_value = value
        for policy in policy_path:
            converted_value = policy(converted_value, **kwargs)

        if not any(isinstance(converted_value, target) for target in target_types):
            pretty_targets = ", ".join(t.__name__ for t in target_types)
            raise MtypeConversionError(
                f"Conversion pipeline did not yield target type {pretty_targets}; "
                f"got {type(converted_value).__name__} instead."
            )

        return converted_value

    def _find_conversion_path(
        self,
        start_type: Type[Any],
        target_types: Tuple[Type[Any], ...],
    ) -> Optional[List[BaseConversionPolicy]]:
        if any(issubclass(start_type, target) for target in target_types):
            return []

        visited: Dict[Type[Any], bool] = {}
        queue: deque[Tuple[Type[Any], List[BaseConversionPolicy]]] = deque()
        queue.append((start_type, []))

        while queue:
            current_type, path = queue.popleft()
            visited[current_type] = True

            if any(issubclass(current_type, target) for target in target_types):
                return path

            for (from_type, to_type), policy in self._policies.items():
                if issubclass(current_type, from_type) and not visited.get(
                    to_type, False
                ):
                    queue.append((to_type, path + [policy]))

        return None

    @classmethod
    def with_default_policies(cls) -> "MtypeConverter":
        return cls(
            policies=[NumpyArrayToPolarsPolicy(), PolarsDataFrameToNumpyPolicy()]
        )


default_converter = MtypeConverter.with_default_policies()


def convert_mtype(
    value: Any, to_type: Union[Type[Any], Tuple[Type[Any], ...]], **kwargs: Any
) -> Any:
    """Convenience function using the module-level default converter."""

    return default_converter.convert(value, to_type, **kwargs)
