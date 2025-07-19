import numpy

from collatable.fields.field import Field
from collatable.fields.sequence_field import SequenceField
from collatable.types import IntTensor


class IndexField(Field[IntTensor]):
    __slots__ = ["_index"]

    def __init__(self, index: int, sequence: SequenceField) -> None:
        if index < 0 or index >= len(sequence):
            raise ValueError(
                f"Index {index} is out of range for sequence of length {len(sequence)}"
            )

        super().__init__(padding_value=-1)
        self._index = index

    def __str__(self) -> str:
        return str(self._index)

    def __repr__(self) -> str:
        return f"IndexField(index={self._index})"

    @property
    def index(self) -> int:
        return self._index

    def as_array(self) -> IntTensor:
        return numpy.array(self.index)

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: IntTensor,
        sequence: SequenceField,
    ) -> "IndexField":
        if array.ndim != 0:
            raise ValueError(
                f"IndexField expects a 0-dimensional array, but got shape {array.shape}"
            )
        return cls(array.item(), sequence)
