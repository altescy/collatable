from typing import Callable, Generic, Hashable, Iterator, Mapping, Optional, Protocol, Sequence, TypeVar, Union, cast

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.types import IntTensor

LabelT = TypeVar("LabelT", bound=Hashable)


class IDecotableIndexer(Protocol[LabelT]):
    def __call__(self, label: LabelT) -> int: ...

    def decode(self, index: int) -> LabelT: ...


class SequenceLabelField(Generic[LabelT], SequenceField[IntTensor]):
    __slots__ = ["_labels", "_indexed_labels"]

    def __init__(
        self,
        labels: Union[Sequence[LabelT]],
        sequence_field: SequenceField,
        *,
        vocab: Optional[Mapping[LabelT, int]] = None,
        indexer: Optional[Callable[[LabelT], int]] = None,
        padding_value: PaddingValue = 0,
    ) -> None:
        if len(labels) != len(sequence_field):
            raise ValueError("SequenceLabelField requires a label for every token in the sequence.")
        if (vocab is None is indexer) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab or indexer.")
        if vocab is not None:
            indexer = self._make_indexer(vocab)

        super().__init__(padding_value=padding_value)

        self._labels = labels
        self._indexed_labels: Sequence[int]
        if isinstance(self._labels[0], int):
            self._indexed_labels = cast(Sequence[int], self._labels)
        else:
            if indexer is None:
                raise ValueError("Indexer must be specified if labels are strings.")
            self._labels = self._labels
            self._indexed_labels = [indexer(label) for label in self._labels]

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[LabelT]:
        return iter(self._labels)

    def __getitem__(self, index: int) -> LabelT:
        return self._labels[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(label) for label in self._labels)}]"

    def __repr__(self) -> str:
        return f"SequenceLabelField(labels={self._labels}, padding_value={self._padding_value})"

    @property
    def labels(self) -> Sequence[LabelT]:
        return self._labels

    def as_array(self) -> IntTensor:
        return numpy.array(self._indexed_labels)

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: IntTensor,
        *,
        sequence_field: SequenceField,
        indexer: Optional[IDecotableIndexer[LabelT]] = None,
    ) -> "SequenceLabelField[LabelT]":
        if array.ndim != 1:
            raise ValueError(f"SequenceLabelField expects a 1-dimensional array, but got shape {array.shape}")
        labels: Sequence[LabelT] = cast(Sequence[LabelT], array.tolist())
        if indexer is not None:
            labels = [indexer.decode(index) for index in array]
        return cls(labels, sequence_field, indexer=indexer)

    @staticmethod
    def _make_indexer(vocab: Mapping[LabelT, int]) -> Callable[[LabelT], int]:
        def indexer(label: LabelT) -> int:
            return vocab[label]

        return indexer
