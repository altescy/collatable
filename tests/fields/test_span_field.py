import numpy

from collatable.extras.indexer import TokenIndexer
from collatable.fields.list_field import ListField
from collatable.fields.span_field import SpanField
from collatable.fields.text_field import TextField


def test_span_field() -> None:
    PAD_TOKEN = "<PAD>"
    indexer = TokenIndexer[str](specials=(PAD_TOKEN,))
    with indexer.context(train=True):
        a = TextField(
            ["embassy", "of", "japan", "in", "the", "united", "states"],
            indexer=indexer,
            padding_value=indexer[PAD_TOKEN],
        )
        b = TextField(
            ["nara", "institute", "of", "science", "and", "technology"],
            indexer=indexer,
            padding_value=indexer[PAD_TOKEN],
        )
        fields = [
            ListField([SpanField(2, 3, a), SpanField(4, 7, a), SpanField(0, 7, a)]),
            ListField([SpanField(0, 1, b), SpanField(0, 6, b)]),
        ]
    output = fields[0].collate(fields)

    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 3, 2)

    span_lengths = (output.max(2) >= 0).sum(1)
    assert span_lengths.tolist() == [3, 2]
