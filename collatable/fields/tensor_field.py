from typing import cast

import numpy

from collatable.fields.field import Field
from collatable.typing import ArrayLike, Tensor


class TensorField(Field[Tensor]):
    __slots__ = ["_tensor", "_padding_value"]

    def __init__(
        self,
        tensor: Tensor,
        padding_value: ArrayLike = 0,
    ) -> None:
        super().__init__(padding_value=padding_value)
        self._tensor = tensor

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    def __eq__(self, other: object) -> bool:
        if isinstance(self, other.__class__):
            other = cast(TensorField, other)
            return numpy.array_equal(self.tensor, other.tensor) and self.padding_value == other.padding_value
        return NotImplemented

    def as_array(self) -> Tensor:
        return self._tensor
