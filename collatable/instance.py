from typing import Iterator

from collatable.fields import Field


class Instance:
    def __init__(self, **fields: Field) -> None:
        self._fields = fields

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __getitem__(self, name: str) -> Field:
        return self._fields[name]
