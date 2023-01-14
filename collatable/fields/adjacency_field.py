import copy
from typing import Callable, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import Tensor

Self = TypeVar("Self", bound="AdjacencyField")


class AdjacencyField(Field[Tensor]):
    __slots__ = ["indices", "labels", "sequence_field", "indexer", "_indexed_labels", "padding_value"]

    def __init__(
        self,
        indices: Sequence[Tuple[int, int]],
        sequence_field: SequenceField,
        *,
        labels: Optional[Union[Sequence[int], Sequence[str]]] = None,
        vocab: Optional[Mapping[str, int]] = None,
        indexer: Optional[Callable[[str], int]] = None,
        padding_value: PaddingValue = -1,
    ) -> None:
        if len(indices) == 0:
            raise ValueError("AdjacencyField requires at least one index.")
        if labels is not None and len(indices) != len(labels):
            raise ValueError("AdjacencyField requires a label for every index.")
        if labels is not None and (vocab is None is indexer) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab or indexer.")
        if labels is not None and isinstance(labels[0], str) and (vocab or indexer) is None:
            raise ValueError("Vocab or indexer must be specified if label is a string.")
        if labels is not None and isinstance(labels[0], str) and vocab is not None:
            indexer = self._make_indexer(vocab)
        if not all(0 <= index < len(sequence_field) for pair in indices for index in pair):
            raise ValueError("Indices must be within the bounds of the sequence.")

        super().__init__(padding_value=padding_value)

        self.indices = indices
        self.labels = labels
        self.sequence_field = sequence_field
        self.indexer = indexer
        if self.labels:
            if isinstance(self.labels[0], int):
                self._indexed_labels = cast(Sequence[int], self.labels)
            else:
                assert self.indexer is not None
                self._indexed_labels = [self.indexer(label) for label in cast(Sequence[str], self.labels)]

    @staticmethod
    def _make_indexer(vocab: Mapping[str, int]) -> Callable[[str], int]:
        def indexer(label: str) -> int:
            return vocab[label]

        return indexer

    def copy(self: Self) -> Self:
        output = self.__class__(
            indices=copy.deepcopy(self.indices),
            sequence_field=self.sequence_field,
            labels=copy.deepcopy(self.labels),
            indexer=self.indexer,
            padding_value=self.padding_value,
        )
        output._indexed_labels = copy.deepcopy(self._indexed_labels)
        return output

    def as_array(self) -> Tensor:
        array = numpy.full(
            (len(self.sequence_field), len(self.sequence_field)),
            self.padding_value[""],
            dtype=numpy.int32,
        )
        labels = self._indexed_labels if self.labels is not None else [1] * len(self.indices)
        for (i, j), label in zip(self.indices, labels):
            array[i, j] = label
        return array
