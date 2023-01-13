from contextlib import contextmanager
from typing import Dict, Generic, Hashable, Iterator, List, Optional, Sequence, TypeVar

import numpy

from collatable.typing import Tensor

T_Value = TypeVar("T_Value", bound=Hashable)
Self = TypeVar("Self", bound="Indexer")


class Indexer(Generic[T_Value]):
    def __init__(
        self,
        *,
        spetials: Sequence[T_Value] = (),
        default: Optional[T_Value] = None,
    ) -> None:
        self._index_to_value: List[T_Value] = list(spetials)
        self._value_to_index: Dict[T_Value, int] = {value: index for index, value in enumerate(spetials)}
        self._default_value = default
        self._train = False

    def __len__(self) -> int:
        return len(self._index_to_value)

    def __getitem__(self, value: T_Value) -> int:
        return self.get_index_by_value(value)

    @contextmanager
    def train(self: Self) -> Iterator[Self]:
        self._train = True
        yield self
        self._train = False

    def get_value_by_index(self, index: int) -> T_Value:
        try:
            return self._index_to_value[index]
        except IndexError:
            if self._default_value is not None:
                return self._default_value
            raise

    def get_index_by_value(self, value: T_Value) -> int:
        if value not in self._value_to_index:
            if not self._train:
                raise KeyError(value)
            self._value_to_index[value] = len(self._index_to_value)
            self._index_to_value.append(value)
        return self._value_to_index[value]


class TokenIndexer(Indexer[str]):
    def __call__(self, tokens: Sequence[str]) -> Dict[str, Tensor]:
        return {
            "token_ids": numpy.array([self.get_index_by_value(value) for value in tokens], dtype=numpy.int64),
            "lengths": numpy.array(len(tokens), dtype=numpy.int64),
        }


class LabelIndexer(Indexer[str]):
    def __call__(self, label: str) -> Tensor:
        return numpy.array(self.get_index_by_value(label), dtype=numpy.int32)
