from contextlib import contextmanager
from typing import (
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
    cast,
)

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
    def context(self, train: bool) -> Iterator[None]:
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

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[T_Value],
        *,
        ignores: Iterable[T_Value] = (),
        specials: Sequence[T_Value] = (),
        default: Optional[T_Value] = None,
    ) -> "Indexer[T_Value]":
        indexer = cls(ignores=ignores, specials=specials, default=default)
        with indexer.context(train=True):
            for value in iterable:
                indexer[value]
        return indexer

    @classmethod
    def from_vocab(cls, vocab: Mapping[T_Value, int]) -> "Indexer[T_Value]":
        class PlaceHolder:
            ...

        placeholder = PlaceHolder()

        indexer: Indexer[T_Value] = cls()
        index_to_value: List[Union[T_Value, PlaceHolder]] = [placeholder] * len(vocab)
        value_to_index: Dict[T_Value, int] = {}
        for value, index in vocab.items():
            if index < 0:
                raise ValueError("index must be non-negative")
            if index >= len(index_to_value):
                raise ValueError("index must be less than the length of the vocabulary")
            if not isinstance(index_to_value[index], PlaceHolder):
                raise ValueError("index must be unique")
            index_to_value[index] = value
            value_to_index[value] = index

        indexer._index_to_value = cast(List[T_Value], index_to_value)
        indexer._value_to_index = value_to_index
        return indexer

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Sequence[T_Value]],
        *,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ignores: Iterable[T_Value] = (),
        specials: Sequence[T_Value] = (),
        default: Optional[T_Value] = None,
    ) -> "Indexer[T_Value]":
        num_documents = 0
        value_to_df: Dict[T_Value, int] = {}
        for tokens in documents:
            num_documents += 1
            for token in tokens:
                value_to_df[token] = value_to_df.get(token, 0) + 1

        min_df = int(min_df) if isinstance(min_df, int) else int(min_df * num_documents)
        max_df = int(max_df) if isinstance(max_df, int) else int(max_df * num_documents)

        indexer = cls(ignores=ignores, specials=specials, default=default)
        with indexer.context(train=True):
            for token, df in list(value_to_df.items()):
                if min_df <= df <= max_df:
                    indexer[token]

        return indexer


class TokenIndexer(Indexer[T_Value]):
    def __call__(self, tokens: Sequence[T_Value]) -> Dict[str, Tensor]:
        return {
            "token_ids": numpy.array([self.get_index_by_value(value) for value in tokens], dtype=numpy.int64),
            "lengths": numpy.array(len(tokens), dtype=numpy.int64),
        }


class LabelIndexer(Indexer[T_Value]):
    def __call__(self, label: T_Value) -> int:
        return self.get_index_by_value(label)
