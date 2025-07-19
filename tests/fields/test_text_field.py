from typing import List

from collatable.fields.text_field import PaddingValue, TextField


def test_text_field_can_be_converted_to_array() -> None:
    tokens = ["this", "is", "a", "test"]
    vocab = {"a": 0, "is": 1, "test": 2, "this": 3}
    field = TextField(tokens, vocab=vocab)
    output = field.as_array()
    assert isinstance(output, dict)
    assert output.keys() == {"token_ids", "mask"}
    assert output["token_ids"].tolist() == [3, 1, 0, 2]
    assert output["mask"].tolist() == [True, True, True, True]


def test_text_field_can_be_collated() -> None:
    vocab = {"a": 0, "first": 1, "is": 2, "this": 3, "second": 4, "sentence": 5, "!": 6}
    padding_value: PaddingValue = {"token_ids": -1}
    fields: List[TextField] = [
        TextField(
            ["this", "is", "a", "first", "sentence"],
            vocab=vocab,
            padding_value=padding_value,
        ),
        TextField(
            ["this", "is", "a", "second", "sentence", "!"],
            vocab=vocab,
            padding_value=padding_value,
        ),
    ]
    output = fields[0].collate(fields)
    assert isinstance(output, dict)
    assert output.keys() == {"token_ids", "mask"}
    assert output["token_ids"].tolist() == [[3, 2, 0, 1, 5, -1], [3, 2, 0, 4, 5, 6]]
    assert output["mask"].sum(1).tolist() == [5, 6]
