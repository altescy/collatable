from typing import Any, List, Sequence

from collatable.fields.field import Field


class MetadataField(Field):
    __slots__ = ["metadata"]

    def __init__(self, metadata: Any) -> None:
        super().__init__()
        self.metadata = metadata

    def as_array(self) -> Any:
        return self.metadata

    def collate(self, arrays: Sequence[Any]) -> List[Any]:
        if isinstance(arrays[0], Field):
            arrays = [array.as_array() for array in arrays]
        return list(arrays)
