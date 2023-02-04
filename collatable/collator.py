from typing import Dict, Sequence

from collatable.instance import Instance
from collatable.typing import DataArray


class Collator:
    def __call__(self, instances: Sequence["Instance"]) -> Dict[str, DataArray]:
        keys = set(instances[0])
        array: Dict[str, DataArray] = {}
        for key in keys:
            values = [instance[key] for instance in instances]
            array[key] = values[0].collate(values)
        return array


def collate(instances: Sequence[Instance]) -> Dict[str, DataArray]:
    return Collator()(instances)
