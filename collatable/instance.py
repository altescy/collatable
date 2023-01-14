from typing import Dict, Iterator, Sequence

from collatable.fields import Field
from collatable.typing import DataArray


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
    def collate(instances: Sequence["Instance"]) -> Dict[str, DataArray]:
        keys = set(instances[0])
        array: Dict[str, DataArray] = {}
        for key in keys:
            values = [instance[key] for instance in instances]
            array[key] = values[0].collate(values)
        return array


def collate(instances: Sequence[Instance]) -> Dict[str, DataArray]:
    return Instance.collate(instances)
