from typing import Callable, Generic, Iterator, Mapping, Optional, Sequence, TypeVar, cast

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import T_DataArray

Token = TypeVar("Token")


class TextField(Generic[Token, T_DataArray], SequenceField[T_DataArray]):
    __slots__ = ["tokens", "indexer"]

    def __init__(
        self,
        tokens: Sequence[Token],
        *,
        vocab: Optional[Mapping[Token, int]] = None,
        indexer: Optional[Callable[[Sequence[Token]], T_DataArray]] = None,
        padding_value: PaddingValue = 0,
    ) -> None:
        if (vocab is None and indexer is None) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab and indexer.")
        if vocab is not None:
            indexer = self._make_indexer(vocab)

        assert indexer is not None

        super().__init__(padding_value=padding_value)

        self.tokens: Sequence[Token] = tokens
        self.indexer: Callable[[Sequence[Token]], T_DataArray] = indexer

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def as_array(self) -> T_DataArray:
        return self.indexer(self.tokens)

    @staticmethod
    def _make_indexer(vocab: Mapping[Token, int]) -> Callable[[Sequence[Token]], T_DataArray]:
        def indexer(tokens: Sequence[Token]) -> T_DataArray:
            output = {
                "token_ids": numpy.array([vocab[token] for token in tokens], dtype=numpy.int64),
                "lengths": numpy.array(len(tokens), dtype=numpy.int64),
            }
            return cast(T_DataArray, output)

        return indexer
