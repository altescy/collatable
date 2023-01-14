from typing import List

import numpy

from collatable.extras.indexer import LabelIndexer, TokenIndexer
from collatable.fields import LabelField, MetadataField, TextField
from collatable.instance import Instance


def test_instance() -> None:
    dataset = [
        ("this is awesome", "positive"),
        ("this is a bad movie", "negative"),
        ("this movie is an awesome movie", "positive"),
        ("this movie is too bad to watch", "negative"),
    ]

    token_indexer = TokenIndexer[str]()
    label_indexer = LabelIndexer[str]()

    instances: List[Instance] = []
    with token_indexer.context(train=True), label_indexer.context(train=True):
        for id_, (text, label) in enumerate(dataset):
            tokens = text.split()
            instance = Instance(
                text=TextField(tokens, indexer=token_indexer),
                label=LabelField(label, indexer=label_indexer),
                metadata=MetadataField({"id": id_}),
            )
            instances.append(instance)

    output = Instance.collate(instances)

    assert isinstance(output, dict)
    assert set(output.keys()) == {"text", "label", "metadata"}
    assert isinstance(output["text"], dict)
    assert isinstance(output["label"], numpy.ndarray)
    assert isinstance(output["metadata"], list)
    assert output["metadata"] == [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]
