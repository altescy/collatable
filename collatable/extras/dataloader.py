import math
import random
from typing import Dict, Iterator, Sequence

from collatable.instance import Instance
from collatable.typing import DataArray


class DataLoader:
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

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._dataset) // self._batch_size
        return math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self) -> Iterator[Dict[str, DataArray]]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.shuffle(indices)
        for i in range(0, len(self._dataset), self._batch_size):
            if self._drop_last and i + self._batch_size > len(self._dataset):
                break
            batch_indices = indices[i : i + self._batch_size]
            batch = [self._dataset[j] for j in batch_indices]
            yield Instance.collate(batch)
