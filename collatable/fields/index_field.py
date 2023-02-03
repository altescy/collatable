import numpy

from collatable.fields.field import Field
from collatable.fields.sequence_field import SequenceField
from collatable.typing import Tensor


class IndexField(Field[Tensor]):
    __slots__ = ["_index"]

    def __init__(self, index: int, sequence: SequenceField) -> None:
        if index < 0 or index >= len(sequence):
            raise ValueError(f"Index {index} is out of range for sequence of length {len(sequence)}")

        super().__init__(padding_value=-1)
        self._index = index

    def __str__(self) -> str:
        return str(self._index)

    def __repr__(self) -> str:
        return f"IndexField(index={self._index})"

    @property
    def index(self) -> int:
        return self._index

    def as_array(self) -> Tensor:
        return numpy.array(self.index)
