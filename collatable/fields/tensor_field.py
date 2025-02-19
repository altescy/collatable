from typing import Generic, cast

import numpy

from collatable.fields.field import Field
from collatable.types import ArrayLike, TensorT


class TensorField(Generic[TensorT], Field[TensorT]):
    __slots__ = ["_tensor", "_padding_value"]

    def __init__(
        self,
        tensor: TensorT,
        padding_value: ArrayLike = 0,
    ) -> None:
        super().__init__(padding_value=padding_value)
        self._tensor = tensor

    @property
    def tensor(self) -> TensorT:
        return self._tensor

    def __eq__(self, other: object) -> bool:
        if isinstance(self, other.__class__):
            other = cast(TensorField, other)
            return numpy.array_equal(self.tensor, other.tensor) and self.padding_value == other.padding_value
        return NotImplemented

    def __str__(self) -> str:
        return str(self._tensor)

    def __repr__(self) -> str:
        return f"TensorField(tensor={self._tensor}, padding_value={self._padding_value})"

    def as_array(self) -> TensorT:
        return self._tensor

    @classmethod
    def from_array(cls, array: TensorT) -> "TensorField":  # type: ignore[override]
        return cls(array)
