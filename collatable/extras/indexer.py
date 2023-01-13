from contextlib import contextmanager
from typing import Dict, Generic, Hashable, Iterable, Iterator, List, Optional, Sequence, Set, TypeVar

import numpy

from collatable.typing import Tensor

T_Value = TypeVar("T_Value", bound=Hashable)
Self = TypeVar("Self", bound="Indexer")


class Indexer(Generic[T_Value]):
    def __init__(
        self,
        *,
        ignores: Iterable[T_Value] = (),
        specials: Sequence[T_Value] = (),
        default: Optional[T_Value] = None,
    ) -> None:
        if default is not None and default not in specials:
            raise ValueError("default value must be in specials")
        self._index_to_value: List[T_Value] = list(specials)
        self._value_to_index: Dict[T_Value, int] = {value: index for index, value in enumerate(specials)}
        self._ignores: Set[T_Value] = set(ignores)
        self._default_value = default
        self._training = False

    def __len__(self) -> int:
        return len(self._index_to_value)

    def __getitem__(self, value: T_Value) -> int:
        return self.get_index_by_value(value)

    @property
    def training(self) -> bool:
        return self._training

    @property
    def freezed(self) -> bool:
        return not self._training

    def train(self) -> None:
        self._training = True

    def frozen(self) -> None:
        self._training = False

    @contextmanager
    def set(self, train: bool) -> Iterator[None]:
        prev_training = self._training
        self._training = train
        try:
            yield
        finally:
            self._training = prev_training

    def get_value_by_index(self, index: int) -> T_Value:
        return self._index_to_value[index]

    def get_index_by_value(self, value: T_Value) -> int:
        if self._default_value is not None and value in self._ignores:
            return self._value_to_index[self._default_value]
        if value not in self._value_to_index:
            if not self._training:
                if self._default_value is None:
                    raise KeyError(value)
                return self._value_to_index[self._default_value]
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
    def __call__(self, label: str) -> int:
        return self.get_index_by_value(label)
