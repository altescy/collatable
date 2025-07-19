from typing import Any, Dict, Mapping, Optional, Sequence, Set

from collatable.fields import Field
from collatable.types import DataArray, INamedTuple


class Collator:
    def __init__(self, field_names: Optional[Set[str]] = None) -> None:
        self._field_names = field_names

    def _extract_fields(self, instance: Any) -> Mapping[str, Field]:
        if not isinstance(instance, Mapping):
            if hasattr(instance, "__dict__"):
                members = instance.__dict__
                slots = set(
                    getattr(
                        instance,
                        "__slots__",
                        [
                            key
                            for key in members
                            if not key.startswith("_")
                            or key in (self._field_names or [])
                        ],
                    )
                )
                if self._field_names is not None and not (self._field_names <= slots):
                    raise ValueError(
                        f"Field names {self._field_names - slots} not found"
                    )
                instance = {slot: members[slot] for slot in slots if slot in members}
            elif isinstance(instance, INamedTuple):
                instance = instance._asdict()
        return {
            key: value
            for key, value in instance.items()
            if isinstance(value, Field)
            and (self._field_names is None or key in self._field_names)
        }

    def __call__(self, instances: Sequence[Any]) -> Dict[str, DataArray]:
        instances = [self._extract_fields(instance) for instance in instances]
        keys = set(next(iter(instances), {}).keys())
        array: Dict[str, DataArray] = {}
        for key in keys:
            values = [instance[key] for instance in instances]
            array[key] = values[0].collate(values)
        return array


def collate(
    instances: Sequence[Any],
    field_names: Optional[Set[str]] = None,
) -> Dict[str, DataArray]:
    return Collator(field_names)(instances)
