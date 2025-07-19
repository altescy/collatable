from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from collatable.fields.field import Field
from collatable.types import DataArray, DataArrayT

Self = TypeVar("Self", bound="Field")
FieldT = TypeVar("FieldT", bound=Field)


class MappingField(
    Generic[DataArrayT, FieldT],
    Field[Dict[str, DataArrayT]],
    Mapping[str, FieldT],
):
    def __init__(self, mapping: Mapping[str, FieldT]) -> None:
        super().__init__()
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __getitem__(self, key: str) -> FieldT:
        return self._mapping[key]

    def __contains__(self, key: object) -> bool:
        return key in self._mapping

    def __str__(self) -> str:
        return str(self._mapping)

    def __repr__(self) -> str:
        return f"MappingField({self._mapping})"

    def as_array(self) -> Dict[str, Any]:
        return {key: field.as_array() for key, field in self._mapping.items()}

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: Mapping[str, DataArray],
        *,
        fields: Mapping[str, Tuple[Type[FieldT], Mapping[str, Any]]],
    ) -> "MappingField":
        return cls(
            {
                key: field.from_array(array[key], **params)
                for key, (field, params) in fields.items()
            }
        )

    def collate(
        self: Self,
        arrays: Union[Sequence[Dict[str, DataArrayT]], Sequence[Self]],
    ) -> Dict[str, DataArrayT]:
        if not isinstance(arrays[0], MappingField):
            arrays = cast(Sequence[Dict[str, DataArrayT]], arrays)
            return super().collate(arrays)
        return {
            key: field.collate(
                [x[key] for x in (x.as_array() for x in cast(Sequence[Self], arrays))]
            )
            for key, field in cast(MappingField, self)._mapping.items()
        }
