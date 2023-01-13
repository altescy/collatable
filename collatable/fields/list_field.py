from typing import Generic, Iterator, Sequence

from collatable.fields.field import Field
from collatable.fields.sequence_field import SequenceField
from collatable.typing import T_DataArray


class ListField(Generic[T_DataArray], SequenceField[T_DataArray]):
    __slots__ = ["fields"]

    def __init__(self, fields: Sequence[Field[T_DataArray]]) -> None:
        super().__init__(padding_value=fields[0].padding_value)
        self.fields: Sequence[Field[T_DataArray]] = fields

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field[T_DataArray]]:
        return iter(self.fields)

    def __getitem__(self, index: int) -> Field[T_DataArray]:
        return self.fields[index]

    def as_array(self) -> T_DataArray:
        return self.fields[0].collate(self.fields)
