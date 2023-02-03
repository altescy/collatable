from typing import Generic, Iterator, Optional, Sequence

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import T_DataArray


class ListField(Generic[T_DataArray], SequenceField[T_DataArray]):
    __slots__ = ["_fields", "_padding_value"]

    def __init__(
        self,
        fields: Sequence[Field[T_DataArray]],
        padding_value: Optional[PaddingValue] = None,
    ) -> None:
        super().__init__(padding_value=padding_value if padding_value is not None else fields[0].padding_value)
        self._fields: Sequence[Field[T_DataArray]] = fields

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field[T_DataArray]]:
        return iter(self.fields)

    def __getitem__(self, index: int) -> Field[T_DataArray]:
        return self.fields[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(field) for field in self._fields)}]"

    def __repr__(self) -> str:
        return f"ListField(fields={self._fields}, padding_value={self._padding_value})"

    @property
    def fields(self) -> Sequence[Field[T_DataArray]]:
        return self._fields

    def as_array(self) -> T_DataArray:
        return self.fields[0].collate(self.fields)
