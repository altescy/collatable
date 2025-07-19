from typing import Generic, TypeVar

from collatable.fields.field import Field
from collatable.types import DataArrayT

Self = TypeVar("Self", bound="SequenceField")


class SequenceField(Generic[DataArrayT], Field[DataArrayT]):
    def __len__(self) -> int:
        raise NotImplementedError
