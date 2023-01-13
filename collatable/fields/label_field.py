from typing import Callable, Mapping, Optional, TypeVar, Union

import numpy

from collatable.fields.field import Field
from collatable.typing import Tensor

Self = TypeVar("Self", bound="LabelField")


class LabelField(Field[Tensor]):
    __slots__ = ["label", "label_index", "indexer"]

    def __init__(
        self,
        label: Union[int, str],
        *,
        vocab: Optional[Mapping[str, int]] = None,
        indexer: Optional[Callable[[str], int]] = None,
    ) -> None:
        if isinstance(label, str) and vocab is None and indexer is None:
            raise ValueError("LabelField with string labels requires vocab or indexer")
        if vocab is not None and indexer is not None:
            raise ValueError("Must specify either vocab or indexer.")
        if isinstance(label, str) and indexer is None:
            assert vocab is not None
            indexer = self._make_indexer(vocab)

        super().__init__()
        self.label = label
        self.indexer = indexer
        if isinstance(label, int):
            self.label_index = label
        else:
            assert indexer is not None
            self.label_index = indexer(label)

    def copy(self: Self) -> Self:
        return self.__class__(label=self.label, indexer=self.indexer)

    def as_array(self) -> Tensor:
        return numpy.array(self.label_index, dtype=numpy.int32)

    @staticmethod
    def _make_indexer(vocab: Mapping[str, int]) -> Callable[[str], int]:
        def indexer(label: str) -> int:
            return vocab[label]

        return indexer
