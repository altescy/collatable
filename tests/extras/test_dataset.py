from pathlib import Path
from typing import Any, Dict, Iterator

from collatable.extras.dataset import Dataset


def test_dataset() -> None:
    def generate_dataset() -> Iterator[Dict[str, Any]]:
        for i in range(100):
            yield {"id": i, "text": f"this is a document {i}"}

    dataset = Dataset.from_iterable(generate_dataset())
    assert len(dataset) == 100
    assert dataset[10] == {"id": 10, "text": "this is a document 10"}
    assert len(dataset[10:20]) == 10

    dataset_path = dataset.path
    del dataset
    assert not dataset_path.exists()


def test_dataset_can_be_restored(tmp_path: Path) -> None:
    def iterator() -> Iterator[str]:
        for i in range(100):
            yield f"this is a document {i}"

    dataset_path = tmp_path / "dataset"
    dataset = Dataset.from_iterable(iterator(), path=dataset_path)
    assert len(dataset) == 100

    del dataset
    assert dataset_path.exists()

    dataset = Dataset.from_path(dataset_path)
    assert len(dataset) == 100
