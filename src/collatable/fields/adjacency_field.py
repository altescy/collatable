from typing import (
    Callable,
    Generic,
    Hashable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import numpy

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.types import IntTensor

Self = TypeVar("Self", bound="AdjacencyField")
LabelT = TypeVar("LabelT", bound=Hashable)


class IDecotableIndexer(Protocol[LabelT]):
    def __call__(self, label: LabelT) -> int: ...

    def decode(self, index: int) -> LabelT: ...


class AdjacencyField(Generic[LabelT], Field[IntTensor]):
    __slots__ = [
        "_indices",
        "_labels",
        "_indexed_labels",
        "_sequence_length",
        "_padding_value",
    ]

    def __init__(
        self,
        indices: Sequence[Tuple[int, int]],
        sequence_field: SequenceField,
        *,
        labels: Optional[Sequence[LabelT]] = None,
        vocab: Optional[Mapping[LabelT, int]] = None,
        indexer: Optional[Callable[[LabelT], int]] = None,
        padding_value: PaddingValue = -1,
    ) -> None:
        if len(indices) == 0:
            raise ValueError("AdjacencyField requires at least one index.")
        if labels is not None and len(indices) != len(labels):
            raise ValueError("AdjacencyField requires a label for every index.")
        if (
            labels is not None
            and (vocab is None is indexer)
            or (vocab is not None and indexer is not None)
        ):
            raise ValueError("Must specify either vocab or indexer.")
        if (
            labels is not None
            and isinstance(labels[0], str)
            and (vocab or indexer) is None
        ):
            raise ValueError("Vocab or indexer must be specified if label is a string.")
        if labels is not None and isinstance(labels[0], str) and vocab is not None:
            indexer = self._make_indexer(vocab)
        if not all(
            0 <= index < len(sequence_field) for pair in indices for index in pair
        ):
            raise ValueError("Indices must be within the bounds of the sequence.")

        super().__init__(padding_value=padding_value)

        self._indices = indices
        self._labels = labels
        self._sequence_length = len(sequence_field)
        self._indexed_labels: Optional[Sequence[int]]
        if self._labels:
            if isinstance(self._labels[0], int):
                self._indexed_labels = cast(Sequence[int], self._labels)
            else:
                assert indexer is not None
                self._indexed_labels = [indexer(label) for label in self._labels]

    def __str__(self) -> str:
        return f"[{', '.join(str(index) for index in self._indices)}]"

    def __repr__(self) -> str:
        return f"AdjacencyField(indices={self._indices}, padding_value={self._padding_value})"

    @staticmethod
    def _make_indexer(vocab: Mapping[LabelT, int]) -> Callable[[LabelT], int]:
        def indexer(label: LabelT) -> int:
            return vocab[label]

        return indexer

    def as_array(self) -> IntTensor:
        array: IntTensor = numpy.full(
            (self._sequence_length, self._sequence_length),
            self.padding_value[""],
            dtype=numpy.int_,
        )
        labels = (
            self._indexed_labels
            if self._labels is not None
            else [1] * len(self._indices)
        )
        assert labels is not None
        for (i, j), label in zip(self._indices, labels):
            array[i, j] = label
        return array

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: IntTensor,
        *,
        sequence_field: SequenceField,
        indexer: Optional[IDecotableIndexer[LabelT]] = None,
        padding_value: PaddingValue = -1,
    ) -> "AdjacencyField[LabelT]":
        if array.ndim != 2:
            raise ValueError(
                f"AdjacencyField expects a 2-dimensional array, but got shape {array.shape}"
            )
        indices = []
        indexed_labels: List[LabelT] = []
        for i, row in enumerate(array):
            for j, label in enumerate(row):
                if label != padding_value:
                    indices.append((i, j))
                    if indexer is not None:
                        indexed_labels.append(indexer.decode(label))

        return cls(
            indices, sequence_field, labels=indexed_labels, padding_value=padding_value
        )
