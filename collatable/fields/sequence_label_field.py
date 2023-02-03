from typing import Callable, Iterator, Mapping, Optional, Sequence, Union, cast

import numpy

from collatable.fields.field import PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import Tensor


class SequenceLabelField(SequenceField[Tensor]):
    __slots__ = ["_labels", "_indexed_labels"]

    def __init__(
        self,
        labels: Union[Sequence[int], Sequence[str]],
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

        self._labels = labels
        self._indexed_labels: Sequence[int]
        if isinstance(self._labels[0], int):
            self._indexed_labels = cast(Sequence[int], self._labels)
        else:
            if indexer is None:
                raise ValueError("Indexer must be specified if labels are strings.")
            self._labels = self._labels
            self._indexed_labels = [indexer(label) for label in cast(Sequence[str], self._labels)]

    def __len__(self) -> int:
        return len(self._labels)

    def __iter__(self) -> Iterator[Union[int, str]]:
        return iter(self._labels)

    def __getitem__(self, index: int) -> Union[int, str]:
        return self._labels[index]

    def __str__(self) -> str:
        return f"[{', '.join(str(label) for label in self._labels)}]"

    def __repr__(self) -> str:
        return f"SequenceLabelField(labels={self._labels}, padding_value={self._padding_value})"

    @property
    def labels(self) -> Union[Sequence[int], Sequence[str]]:
        return self._labels

    def as_array(self) -> Tensor:
        return numpy.array(self._indexed_labels)

    @staticmethod
    def _make_indexer(vocab: Mapping[str, int]) -> Callable[[str], int]:
        def indexer(label: str) -> int:
            return vocab[label]

        return indexer
