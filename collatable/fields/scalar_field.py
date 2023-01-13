from typing import Optional

import numpy

from collatable.fields.field import Field
from collatable.typing import T_Scalar, Tensor
from collatable.util import get_scalar_default_value


class ScalarField(Field[Tensor]):
    __slots__ = ["value", "padding_value"]

    def __init__(
        self,
        value: T_Scalar,
        padding_value: Optional[T_Scalar] = None,
    ) -> None:
        if padding_value is None:
            padding_value = get_scalar_default_value(type(value))
        super().__init__(padding_value=padding_value)
        self.value: T_Scalar = value

    def as_array(self) -> Tensor:
        return numpy.array(self.value)
