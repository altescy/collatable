from typing import Callable, Generic, Hashable, Mapping, Optional, TypeVar

import numpy

from collatable.fields.field import Field
from collatable.typing import IntTensor

Self = TypeVar("Self", bound="LabelField")
LabelT = TypeVar("LabelT", bound=Hashable)


class LabelField(Generic[LabelT], Field[IntTensor]):
    __slots__ = ["_label", "_label_index"]

    def __init__(
        self,
        label: LabelT,
        *,
        vocab: Optional[Mapping[LabelT, int]] = None,
        indexer: Optional[Callable[[LabelT], int]] = None,
    ) -> None:
        if isinstance(label, str) and vocab is None is indexer:
            raise ValueError("LabelField with string labels requires vocab or indexer")
        if vocab is not None and indexer is not None:
            raise ValueError("Must specify either vocab or indexer.")
        if isinstance(label, str) and indexer is None:
            assert vocab is not None
            indexer = self._make_indexer(vocab)

        super().__init__()
        self._label = label
        self._label_index: int
        if isinstance(label, int):
            self._label_index = label
        else:
            assert indexer is not None
            self._label_index = indexer(label)

    def __str__(self) -> str:
        return str(self._label)

    def __repr__(self) -> str:
        return f"LabelField(label={self._label})"

    @property
    def label(self) -> LabelT:
        return self._label

    def as_array(self) -> IntTensor:
        return numpy.array(self._label_index, dtype=numpy.int32)

    @staticmethod
    def _make_indexer(vocab: Mapping[LabelT, int]) -> Callable[[LabelT], int]:
        def indexer(label: LabelT) -> int:
            return vocab[label]

        return indexer
