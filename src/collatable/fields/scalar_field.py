from typing import Optional

import numpy

from collatable.fields.field import Field
from collatable.types import ScalarT, Tensor
from collatable.utils import get_scalar_default_value


class ScalarField(Field[Tensor]):
    __slots__ = ["_value", "_padding_value"]

    def __init__(
        self,
        value: ScalarT,
        padding_value: Optional[ScalarT] = None,
    ) -> None:
        if padding_value is None:
            padding_value = get_scalar_default_value(type(value))
        super().__init__(padding_value=padding_value)
        self._value = value

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return f"ScalarField(value={self._value}, padding_value={self._padding_value})"

    def as_array(self) -> Tensor:
        return numpy.array(self._value)

    @classmethod
    def from_array(cls, array: Tensor) -> "ScalarField":  # type: ignore[override]
        return cls(array.item())
