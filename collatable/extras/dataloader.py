import math
import random
from typing import Dict, Iterator, Sequence

from collatable.collator import Collator
from collatable.instance import Instance
from collatable.typing import DataArray


class BatchIterator:
    def __init__(
        self,
        dataset: Sequence[Instance],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._offset = 0
        self._collator = Collator()
        self._indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.shuffle(self._indices)

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._dataset) // self._batch_size
        return math.ceil(len(self._dataset) / self._batch_size)

    def __next__(self) -> Dict[str, DataArray]:
        if self._offset >= len(self._dataset):
            raise StopIteration
        if self._offset + self._batch_size > len(self._dataset):
            if self._drop_last:
                raise StopIteration
            batch_indices = self._indices[self._offset :]
        else:
            batch_indices = self._indices[self._offset : self._offset + self._batch_size]
        self._offset += self._batch_size
        return self._collator([self._dataset[i] for i in batch_indices])

    def __iter__(self) -> Iterator[Dict[str, DataArray]]:
        return self


class DataLoader:
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._collator = Collator()

    def __call__(self, dataset: Sequence[Instance]) -> BatchIterator:
        return BatchIterator(
            dataset,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
        )
