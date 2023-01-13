from typing import Dict, Iterator, Sequence

from collatable.fields import Field
from collatable.typing import Tensor


class Instance:
    def __init__(self, **fields: Field) -> None:
        self._fields = fields

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __getitem__(self, name: str) -> Field:
        return self._fields[name]

    @staticmethod
    def collate(instances: Sequence["Instance"]) -> Dict[str, Tensor]:
        keys = set(instances[0])
        array: Dict[str, Tensor] = {}
        for key in keys:
            values = [instance[key] for instance in instances]
            array[key] = values[0].collate(values)
        return array


def collate(instances: Sequence[Instance]) -> Dict[str, Tensor]:
    return Instance.collate(instances)
