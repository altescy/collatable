from typing import TypeVar

from collatable.fields.field import Field
from collatable.typing import T_DataArray

Self = TypeVar("Self", bound="SequenceField")


class SequenceField(Field[T_DataArray]):
    def __len__(self) -> int:
        raise NotImplementedError
