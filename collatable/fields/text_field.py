from typing import Callable, Generic, Hashable, Iterator, Mapping, Optional, Protocol, Sequence, TypeVar

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField

Self = TypeVar("Self", bound="TextField")
TokenT = TypeVar("TokenT", bound=Hashable)


class IDecotableIndexer(Protocol[TokenT]):
    def __call__(self, tokens: Sequence[TokenT]) -> Mapping[str, numpy.ndarray]: ...

    def decode(self, index: Mapping[str, numpy.ndarray]) -> Sequence[TokenT]: ...


class TextField(Generic[TokenT], SequenceField[Mapping[str, numpy.ndarray]]):
    __slots__ = ["_tokens", "_padding_value", "_indexed_tokens"]

    def __init__(
        self,
        tokens: Sequence[TokenT],
        *,
        vocab: Optional[Mapping[TokenT, int]] = None,
        indexer: Optional[Callable[[Sequence[TokenT]], Mapping[str, numpy.ndarray]]] = None,
        padding_value: PaddingValue = 0,
    ) -> None:
        if (vocab is None is indexer) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab or indexer.")
        if vocab is not None:
            indexer = self._make_indexer(vocab)

        assert indexer is not None

        super().__init__(padding_value=padding_value)

        self._tokens = tokens
        self._indexed_tokens = indexer(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[TokenT]:
        return iter(self.tokens)

    def __getitem__(self, index: int) -> TokenT:
        return self.tokens[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(token) for token in self.tokens)}]"

    def __repr__(self) -> str:
        return f"TextField(tokens={self.tokens}, padding_value={self._padding_value})"

    @property
    def tokens(self) -> Sequence[TokenT]:
        return self._tokens

    def as_array(self) -> Mapping[str, numpy.ndarray]:
        return self._indexed_tokens

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: Mapping[str, numpy.ndarray],
        *,
        indexer: IDecotableIndexer[TokenT],
        padding_value: PaddingValue = 0,
    ) -> "TextField":
        tokens = indexer.decode(array)
        return cls(tokens, indexer=indexer, padding_value=padding_value)

    @staticmethod
    def _make_indexer(vocab: Mapping[TokenT, int]) -> Callable[[Sequence[TokenT]], Mapping[str, numpy.ndarray]]:
        def indexer(tokens: Sequence[TokenT]) -> Mapping[str, numpy.ndarray]:
            token_ids: numpy.ndarray = numpy.array([vocab[token] for token in tokens], dtype=numpy.int64)
            mask: numpy.ndarray = numpy.ones_like(token_ids, dtype=bool)
            output = {"token_ids": token_ids, "mask": mask}
            return output

        return indexer
