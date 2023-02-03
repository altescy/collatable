import abc
import copy
from typing import Dict, Generic, List, Sequence, TypeVar, Union, cast

import numpy

from collatable.typing import ArrayLike, T_DataArray
from collatable.util import stack_with_padding

Self = TypeVar("Self", bound="Field")
PaddingValue = Union[Dict[str, ArrayLike], ArrayLike]


class Field(abc.ABC, Generic[T_DataArray]):
    __slots__: List[str]

    def __init__(self, padding_value: PaddingValue = 0) -> None:
        if not isinstance(padding_value, dict):
            padding_value = {"": padding_value}

        self._padding_value = padding_value

    def __eq__(self: Self, other: object) -> bool:
        if isinstance(self, other.__class__):
            for cls in self.__class__.mro():
                for attr in getattr(cls, "__slots__", []):
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            if hasattr(self, "__dict__"):
                return self.__dict__ == other.__dict__
            return True
        return NotImplemented

    @property
    def padding_value(self) -> Dict[str, ArrayLike]:
        return self._padding_value

    def collate(self: Self, arrays: Union[Sequence[T_DataArray], Sequence[Self]]) -> T_DataArray:
        if isinstance(arrays[0], Field):
            arrays = [cast(Self, array).as_array() for array in arrays]
        arrays = cast(Sequence[T_DataArray], arrays)
        if isinstance(arrays[0], numpy.ndarray):
            return stack_with_padding(arrays, padding_value=self.padding_value[""])
        if isinstance(arrays[0], list):
            return list(arrays)
        if isinstance(arrays[0], dict):
            return {
                key: stack_with_padding(
                    [array[key] for array in arrays],  # type: ignore
                    padding_value=self.padding_value.get(key, 0),
                )
                for key in arrays[0]
            }
        raise TypeError(f"Unsupported type: {type(arrays[0])}")

    def copy(self: Self) -> Self:
        return copy.deepcopy(self)

    @abc.abstractmethod
    def as_array(self) -> T_DataArray:
        raise NotImplementedError
