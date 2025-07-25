from typing import Iterator

from collatable import LabelField, MetadataField, TextField
from collatable.extras.dataloader import DataLoader, DefaultBatchSampler
from collatable.extras.dataset import Dataset
from collatable.extras.indexer import LabelIndexer, TokenIndexer


def test_dataloader() -> None:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    token_indexer = TokenIndexer[str](
        specials=[PAD_TOKEN, UNK_TOKEN], default=UNK_TOKEN
    )
    label_indexer = LabelIndexer[str]()

    def read_dataset() -> Iterator[dict]:
        dataset = [
            ("this is awesome", "positive"),
            ("this is a bad movie", "negative"),
            ("this movie is an awesome movie", "positive"),
            ("this movie is too bad to watch", "negative"),
        ]

        with token_indexer.context(train=True), label_indexer.context(train=True):
            for id_, (text, label) in enumerate(dataset):
                # Prepare each field with the corresponding field class
                text_field = TextField(
                    text.split(),
                    indexer=token_indexer,
                    padding_value=token_indexer[PAD_TOKEN],
                )
                label_field = LabelField(
                    label,
                    indexer=label_indexer,
                )
                metadata_field = MetadataField({"id": id_})
                # Combine these fields into instance
                instance = dict(
                    text=text_field,
                    label=label_field,
                    metadata=metadata_field,
                )
                yield instance

    dataset = Dataset.from_iterable(read_dataset())

    dataloader = DataLoader(DefaultBatchSampler(batch_size=2))
    batch_iterator = dataloader(dataset)
    assert len(batch_iterator) == 2

    dataloader = DataLoader(DefaultBatchSampler(batch_size=3, drop_last=True))
    batch_iterator = dataloader(dataset)
    assert len(batch_iterator) == 1

    dataloader = DataLoader(DefaultBatchSampler(batch_size=2, shuffle=True))
    batch_iterator = dataloader(dataset)
    assert all(len(batch["label"]) == 2 for batch in batch_iterator)
