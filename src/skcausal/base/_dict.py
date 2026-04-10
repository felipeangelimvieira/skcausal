__all__ = ["_StrictDict"]


class _StrictDict(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._fields = list(self.keys())

    def __setitem__(self, key, value):
        if key not in self._fields:
            raise ValueError(
                f"Invalid key '{key}' in {self.__class__.__name__}. "
                f"Allowed keys are: {self._fields}."
            )
        super().__setitem__(key, value)
