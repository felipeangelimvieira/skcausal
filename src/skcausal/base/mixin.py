from typing import Any, TYPE_CHECKING
from skcausal.datatypes import convert, collect_column_types
from skcausal.datatypes._typing import DataFrameLike


class TreatmentCheckMixin:
    """
    Common checks for treatment dataframes.

    Should be used with skbase objects that have "backend"
    and "capability:t_type" tags,
    and optionally "capability:multidimensional_treatment".
    """

    def _check_and_transform_t(self, t: DataFrameLike, *, is_fit: bool = False):
        """
        Preprocess the treatment dataframe.

        This method is called during fit and predict times.
        Does the following checks and actions:

        * Convert pl.Enum columns to 1-hot encoded columns if
            self.get_tag("one_hot_encode_enum_columns", False) is True.
        * Asserts that the schema (after preprocessing)
            is the same as the first time that the method was called.


        """

        t = convert(t, self.get_tag("backend"))

        metadata = {}
        metadata["t_column_types"] = collect_column_types(t)
        self._assert_t_metadata_valid(metadata)

        if is_fit:
            self._t_metadata = metadata
        else:
            if metadata["t_column_types"] != self._t_metadata["t_column_types"]:
                raise ValueError(
                    f"Treatment data schema at predict time does not match fit time. "
                    f"Expected {self._t_metadata['t_column_types']}, "
                    f"got {metadata['t_column_types']}."
                )

        return t

    def _assert_t_metadata_valid(self, metadata: dict[str, Any]):
        messages = []

        # Check t types
        unsupported_type_messages = {}
        for column_name, column_type in metadata["t_column_types"].items():
            if column_type not in self.get_tag("capability:t_type"):
                unsupported_type_messages[column_type] = unsupported_type_messages.get(
                    column_type, []
                ) + [column_name]

        for column_type, column_names in unsupported_type_messages.items():
            messages.append(
                f"Treatment columns {column_names} are of type {column_type}, which is not supported by this estimator."
            )

        # Check multi-dimensionality
        if (
            not self.get_tag("capability:multidimensional_treatment", False)
            and len(metadata["t_column_types"]) > 1
        ):
            messages.append(
                f"Treatment data has columns {list(metadata['t_column_types'].keys())}, but this estimator does not support multidimensional treatment."
            )
        if messages:
            raise ValueError("Error while asserting metadata.\n" + "\n".join(messages))
