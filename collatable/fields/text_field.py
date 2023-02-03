from typing import Callable, Generic, Iterator, Mapping, Optional, Sequence, TypeVar, cast

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import T_DataArray

Self = TypeVar("Self", bound="TextField")
Token = TypeVar("Token")


class TextField(Generic[Token, T_DataArray], SequenceField[T_DataArray]):
    __slots__ = ["_tokens", "_padding_value", "_indexed_tokens"]

    def __init__(
        self,
        tokens: Sequence[Token],
        *,
        vocab: Optional[Mapping[Token, int]] = None,
        indexer: Optional[Callable[[Sequence[Token]], T_DataArray]] = None,
        padding_value: PaddingValue = 0,
    ) -> None:
        if (vocab is None is indexer) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab or indexer.")
        if vocab is not None:
            indexer = self._make_indexer(vocab)

        assert indexer is not None

        super().__init__(padding_value=padding_value)

        self._tokens = tokens
        self._indexed_tokens: T_DataArray = indexer(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)

    def __getitem__(self, index: int) -> Token:
        return self.tokens[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(token) for token in self.tokens)}]"

    def __repr__(self) -> str:
        return f"TextField(tokens={self.tokens}, padding_value={self._padding_value})"

    @property
    def tokens(self) -> Sequence[Token]:
        return self._tokens

    def as_array(self) -> T_DataArray:
        return self._indexed_tokens

    @staticmethod
    def _make_indexer(vocab: Mapping[Token, int]) -> Callable[[Sequence[Token]], T_DataArray]:
        def indexer(tokens: Sequence[Token]) -> T_DataArray:
            token_ids = numpy.array([vocab[token] for token in tokens], dtype=numpy.int64)
            mask = numpy.ones_like(token_ids, dtype=bool)
            output = {"token_ids": token_ids, "mask": mask}
            return cast(T_DataArray, output)

        return indexer
