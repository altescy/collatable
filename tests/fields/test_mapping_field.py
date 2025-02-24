from typing import Mapping, Sequence

import numpy

from collatable.fields import LabelField, MappingField, TextField
from collatable.fields.text_field import PaddingValue
from collatable.utils import debatched


def test_mapping_field_can_be_converted_to_array() -> None:
    tokens = ["this", "is", "a", "test"]
    vocab = {"a": 0, "is": 1, "test": 2, "this": 3}
    text_field = TextField(tokens, vocab=vocab)

    labels = {"negative": 0, "positive": 1}
    label_field = LabelField("positive", vocab=labels)

    field = MappingField({"text": text_field, "label": label_field})
    array = field.as_array()

    assert isinstance(array, dict)
    assert array.keys() == {"text", "label"}
    assert array["text"]["token_ids"].tolist() == [3, 1, 0, 2]
    assert array["text"]["mask"].tolist() == [True, True, True, True]
    assert array["label"] == 1


def test_mapping_field_can_be_collated() -> None:
    class TokenIndexer:
        VOCAB = {"<pad>": -1, "a": 0, "first": 1, "is": 2, "this": 3, "second": 4, "sentence": 5, "!": 6}
        INV_VOCAB = {index: token for token, index in VOCAB.items()}

        def __call__(self, tokens: Sequence[str], /) -> Mapping[str, numpy.ndarray]:
            return {
                "token_ids": numpy.array([self.VOCAB[token] for token in tokens], dtype=numpy.int64),
                "mask": numpy.array([True] * len(tokens), dtype=numpy.bool_),
            }

        def decode(self, index: Mapping[str, numpy.ndarray], /) -> Sequence[str]:
            return [self.INV_VOCAB[index] for index in index["token_ids"] if index != -1]

    class LabelIndexer:
        LABELS = {"negative": 0, "positive": 1}
        INV_LABELS = {index: label for label, index in LABELS.items()}

        def __call__(self, label: str, /) -> int:
            return self.LABELS[label]

        def decode(self, index: int, /) -> str:
            return self.INV_LABELS[index]

    token_indexer = TokenIndexer()
    label_indexer = LabelIndexer()
    padding_value: PaddingValue = {"token_ids": -1}
    fields = [
        MappingField(
            {
                "text": TextField(
                    ["this", "is", "a", "first", "sentence"], indexer=token_indexer, padding_value=padding_value
                ),
                "label": LabelField("positive", indexer=label_indexer),
            }
        ),
        MappingField(
            {
                "text": TextField(
                    ["this", "is", "a", "second", "sentence", "!"], indexer=token_indexer, padding_value=padding_value
                ),
                "label": LabelField("negative", indexer=label_indexer),
            }
        ),
    ]

    output = fields[0].collate(fields)

    assert isinstance(output, dict)
    assert output.keys() == {"text", "label"}
    assert output["text"]["token_ids"].tolist() == [[3, 2, 0, 1, 5, -1], [3, 2, 0, 4, 5, 6]]
    assert output["text"]["mask"].sum(1).tolist() == [5, 6]
    assert output["label"].tolist() == [1, 0]

    reconstruction = [
        MappingField.from_array(
            array,  # type: ignore[arg-type]
            fields={
                "text": (TextField, {"indexer": token_indexer}),
                "label": (LabelField, {"indexer": label_indexer}),
            },
        )
        for array in debatched(output)
    ]

    assert len(reconstruction) == len(fields)
    assert all(isinstance(field, MappingField) for field in reconstruction)
    assert isinstance(reconstruction[0]["text"], TextField)
    assert isinstance(reconstruction[1]["text"], TextField)
    assert reconstruction[0]["text"].tokens == ["this", "is", "a", "first", "sentence"]
    assert reconstruction[1]["text"].tokens == ["this", "is", "a", "second", "sentence", "!"]
    assert isinstance(reconstruction[0]["label"], LabelField)
    assert isinstance(reconstruction[1]["label"], LabelField)
    assert reconstruction[0]["label"].label == "positive"
    assert reconstruction[1]["label"].label == "negative"
