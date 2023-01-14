from typing import Callable, Mapping, Optional, Sequence, Union, cast

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import Tensor


class SequenceLabelField(SequenceField[Tensor]):
    def __init__(
        self,
        labels: Union[Sequence[str], Sequence[int]],
        sequence_field: SequenceField,
        *,
        vocab: Optional[Mapping[str, int]] = None,
        indexer: Optional[Callable[[str], int]] = None,
        padding_value: PaddingValue = 0,
    ) -> None:
        if len(labels) != len(sequence_field):
            raise ValueError("SequenceLabelField requires a label for every token in the sequence.")
        if (vocab is None is indexer) or (vocab is not None and indexer is not None):
            raise ValueError("Must specify either vocab or indexer.")
        if vocab is not None:
            indexer = self._make_indexer(vocab)

        super().__init__(padding_value=padding_value)

        self.labels: Union[Sequence[str], Sequence[int]] = labels
        self.indexer = indexer
        self.sequence_field = sequence_field
        self._indexed_labels: Sequence[int]
        if isinstance(self.labels[0], int):
            self._indexed_labels = cast(Sequence[int], self.labels)
        else:
            if self.indexer is None:
                raise ValueError("Indexer must be specified if labels are strings.")
            self.labels = cast(Sequence[str], self.labels)
            self._indexed_labels = [self.indexer(label) for label in self.labels]

    @staticmethod
    def _make_indexer(vocab: Mapping[str, int]) -> Callable[[str], int]:
        def indexer(label: str) -> int:
            return vocab[label]

        return indexer

    def as_array(self) -> Tensor:
        return numpy.array(self._indexed_labels)
