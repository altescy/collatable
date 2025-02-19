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

ValueT = TypeVar("ValueT", bound=Hashable)
Self = TypeVar("Self", bound="Indexer")


class Indexer(Generic[ValueT]):
    def __init__(
        self,
        *,
        ignores: Iterable[ValueT] = (),
        specials: Sequence[ValueT] = (),
        bos: Optional[ValueT] = None,
        eos: Optional[ValueT] = None,
        default: Optional[ValueT] = None,
    ) -> None:
        if bos is not None and bos not in specials:
            raise ValueError("bos value must be in specials")
        if eos is not None and eos not in specials:
            raise ValueError("eos value must be in specials")
        if default is not None and default not in specials:
            raise ValueError("default value must be in specials")

        self._index_to_value: List[ValueT] = list(specials)
        self._value_to_index: Dict[ValueT, int] = {value: index for index, value in enumerate(specials)}
        self._ignores: Set[ValueT] = set(ignores)
        self._bos_value = bos
        self._eos_value = eos
        self._default_value = default
        self._training = False

    def __len__(self) -> int:
        return len(self._index_to_value)

    def __getitem__(self, value: ValueT) -> int:
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

    def get_value_by_index(self, index: int) -> ValueT:
        return self._index_to_value[index]

    def get_index_by_value(self, value: ValueT) -> int:
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
        iterable: Iterable[ValueT],
        *,
        ignores: Iterable[ValueT] = (),
        specials: Sequence[ValueT] = (),
        bos: Optional[ValueT] = None,
        eos: Optional[ValueT] = None,
        default: Optional[ValueT] = None,
    ) -> "Indexer[ValueT]":
        indexer = cls(ignores=ignores, specials=specials, bos=bos, eos=eos, default=default)
        with indexer.context(train=True):
            for value in iterable:
                indexer[value]
        return indexer

    @classmethod
    def from_vocab(cls, vocab: Mapping[ValueT, int]) -> "Indexer[ValueT]":
        class PlaceHolder: ...

        placeholder = PlaceHolder()

        indexer: Indexer[ValueT] = cls()
        index_to_value: List[Union[ValueT, PlaceHolder]] = [placeholder] * len(vocab)
        value_to_index: Dict[ValueT, int] = {}
        for value, index in vocab.items():
            if index < 0:
                raise ValueError("index must be non-negative")
            if index >= len(index_to_value):
                raise ValueError("index must be less than the length of the vocabulary")
            if not isinstance(index_to_value[index], PlaceHolder):
                raise ValueError("index must be unique")
            index_to_value[index] = value
            value_to_index[value] = index

        indexer._index_to_value = cast(List[ValueT], index_to_value)
        indexer._value_to_index = value_to_index
        return indexer

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Sequence[ValueT]],
        *,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ignores: Iterable[ValueT] = (),
        specials: Sequence[ValueT] = (),
        bos: Optional[ValueT] = None,
        eos: Optional[ValueT] = None,
        default: Optional[ValueT] = None,
    ) -> "Indexer[ValueT]":
        num_documents = 0
        value_to_df: Dict[ValueT, int] = {}
        for tokens in documents:
            num_documents += 1
            for token in set(tokens):
                value_to_df[token] = value_to_df.get(token, 0) + 1

        min_df = int(min_df) if isinstance(min_df, int) else int(min_df * num_documents)
        max_df = int(max_df) if isinstance(max_df, int) else int(max_df * num_documents)

        indexer = cls(ignores=ignores, specials=specials, bos=bos, eos=eos, default=default)
        with indexer.context(train=True):
            for token, df in list(value_to_df.items()):
                if min_df <= df <= max_df:
                    indexer[token]

        return indexer


class TokenIndexer(Generic[ValueT], Indexer[ValueT]):
    def encode(self, tokens: Sequence[ValueT]) -> Mapping[str, Tensor]:
        token_ids = [self.get_index_by_value(value) for value in tokens]
        if self._bos_value is not None:
            token_ids = [self._value_to_index[self._bos_value]] + token_ids
        if self._eos_value is not None:
            token_ids = token_ids + [self._value_to_index[self._eos_value]]
        return {"token_ids": numpy.array(token_ids, dtype=numpy.int64), "mask": numpy.ones(len(token_ids), dtype=bool)}

    def decode(self, index: Mapping[str, Tensor]) -> Sequence[ValueT]:
        token_ids = index["token_ids"]
        mask = index["mask"]
        return [self.get_value_by_index(token_id) for token_id, m in zip(token_ids, mask) if m]

    def __call__(self, tokens: Sequence[ValueT]) -> Mapping[str, Tensor]:
        return self.encode(tokens)


class LabelIndexer(Generic[ValueT], Indexer[ValueT]):
    def encode(self, label: ValueT) -> int:
        return self.get_index_by_value(label)

    def decode(self, index: int) -> ValueT:
        return self.get_value_by_index(index)

    def __call__(self, label: ValueT) -> int:
        return self.encode(label)
