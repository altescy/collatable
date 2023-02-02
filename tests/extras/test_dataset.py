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
