from typing import Generic, Iterator, Optional, Sequence, Type

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import DataArrayT


class ListField(Generic[DataArrayT], SequenceField[DataArrayT]):
    __slots__ = ["_fields", "_padding_value"]

    def __init__(
        self,
        fields: Sequence[Field[DataArrayT]],
        padding_value: Optional[PaddingValue] = None,
    ) -> None:
        super().__init__(padding_value=padding_value if padding_value is not None else fields[0].padding_value)
        self._fields: Sequence[Field[DataArrayT]] = fields

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self) -> Iterator[Field[DataArrayT]]:
        return iter(self.fields)

    def __getitem__(self, index: int) -> Field[DataArrayT]:
        return self.fields[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(field) for field in self._fields)}]"

    def __repr__(self) -> str:
        return f"ListField(fields={self._fields}, padding_value={self._padding_value})"

    @property
    def fields(self) -> Sequence[Field[DataArrayT]]:
        return self._fields

    def as_array(self) -> DataArrayT:
        return self.fields[0].collate(self.fields)

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: DataArrayT,
        *,
        item_type: Type[Field],
        padding_value: Optional[PaddingValue] = None,
    ) -> "ListField":
        return cls([item_type.from_array(item) for item in array], padding_value=padding_value)
