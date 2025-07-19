import numpy

from collatable.extras.indexer import TokenIndexer
from collatable.fields.sequence_label_field import SequenceLabelField
from collatable.fields.text_field import TextField


def test_sequence_label_field() -> None:
    label_vocab = {"O": 0, "B": 1, "I": 2, "U": 3}
    dataset = [
        (["O", "O", "O", "B", "I"], ["my", "name", "is", "john", "smith"]),
        (
            ["O", "O", "O", "U", "O", "O", "O"],
            ["i", "lived", "in", "japan", "three", "years", "ago"],
        ),
    ]
    token_indexer = TokenIndexer[str]()
    fields = []
    with token_indexer.context(train=True):
        for labels, tokens in dataset:
            text_field = TextField(tokens, indexer=token_indexer)
            sequence_label_field = SequenceLabelField(
                labels, text_field, vocab=label_vocab
            )
            fields.append(sequence_label_field)

    output = fields[0].collate(fields)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 7)
    assert output[0].tolist() == [0, 0, 0, 1, 2, 0, 0]
    assert output[1].tolist() == [0, 0, 0, 3, 0, 0, 0]
