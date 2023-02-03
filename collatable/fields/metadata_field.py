from typing import Any, List, Sequence

from collatable.fields.field import Field


class MetadataField(Field):
    __slots__ = ["_metadata"]

    def __init__(self, metadata: Any) -> None:
        super().__init__()
        self._metadata = metadata

    def __str__(self) -> str:
        return str(self._metadata)

    def __repr__(self) -> str:
        return f"MetadataField(metadata={self._metadata})"

    @property
    def metadata(self) -> Any:
        return self._metadata

    def as_array(self) -> Any:
        return self._metadata

    def collate(self, arrays: Sequence[Any]) -> List[Any]:
        if isinstance(arrays[0], Field):
            arrays = [array.as_array() for array in arrays]
        return list(arrays)
