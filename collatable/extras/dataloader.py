import math
import random
from typing import Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence, TypeVar

from collatable.collator import Collator
from collatable.fields import Field
from collatable.types import DataArray

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class SizedIterator(Protocol[T_co]):
    def __len__(self) -> int: ...

    def __next__(self) -> T_co: ...

    def __iter__(self) -> Iterator[T_co]: ...


class BatchIterator(SizedIterator[Dict[str, DataArray]]):
    def __init__(
        self,
        dataset: Sequence[Mapping[str, Field]],
        indices: Iterable[Sequence[int]],
        num_batches: int,
        collator: Optional[Collator] = None,
    ) -> None:
        self._dataset = dataset
        self._indices = iter(indices)
        self._num_batches = num_batches
        self._collator = collator or Collator()

    def __len__(self) -> int:
        return self._num_batches

    def __next__(self) -> Dict[str, DataArray]:
        indices = next(self._indices)
        return self._collator([self._dataset[i] for i in indices])

    def __iter__(self) -> Iterator[Dict[str, DataArray]]:
        return self


class IBatchSampler(Protocol):
    def __call__(self, dataset: Sequence) -> SizedIterator[Mapping[str, DataArray]]: ...


class DefaultBatchSampler:
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __call__(self, dataset: Sequence) -> BatchIterator:
        num_batches = (
            len(dataset) // self._batch_size if self._drop_last else math.ceil(len(dataset) / self._batch_size)
        )
        indices = list(range(len(dataset)))
        if self._shuffle:
            random.shuffle(indices)

        def iter_batches() -> Iterator[Sequence[int]]:
            for batch_index in range(num_batches):
                start_index = batch_index * self._batch_size
                end_index = start_index + self._batch_size
                yield indices[start_index:end_index]

        return BatchIterator(dataset, iter_batches(), num_batches)


class DataLoader:
    def __init__(
        self,
        sampler: Optional[IBatchSampler] = None,
        collator: Optional[Collator] = None,
    ) -> None:
        self._sampler = sampler or DefaultBatchSampler()
        self._collator = collator or Collator()

    def __call__(self, dataset: Sequence[Mapping[str, Field]]) -> SizedIterator[Mapping[str, DataArray]]:
        return self._sampler(dataset)
