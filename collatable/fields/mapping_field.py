from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple, Type

from collatable.fields.field import Field
from collatable.types import DataArray


class MappingField(Field[Dict[str, Any]], Mapping[str, Field]):
    def __init__(self, mapping: Mapping[str, Field]) -> None:
        super().__init__()
        self._mapping = mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __getitem__(self, key: str) -> Field:
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
        fields: Mapping[str, Tuple[Type[Field], Mapping[str, Any]]],
    ) -> "MappingField":
        return cls({key: field.from_array(array[key], **params) for key, (field, params) in fields.items()})

    def collate(  # type: ignore[override]
        self,
        arrays: Sequence,
    ) -> Dict[str, Any]:
        if not isinstance(arrays[0], MappingField):
            return super().collate(arrays)  # type: ignore[no-any-return]
        arrays = [x.as_array() for x in arrays]
        return {key: field.collate([x[key] for x in arrays]) for key, field in self._mapping.items()}
