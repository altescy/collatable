from collatable.fields.label_field import LabelField


def test_label_field_can_be_converted_to_array() -> None:
    field = LabelField(2)
    output = field.as_array()
    assert output == 2


def test_label_field_can_be_converted_to_array_with_indexer() -> None:
    vocab = {"a": 0, "b": 1, "c": 2}
    field = LabelField("b", vocab=vocab)
    output = field.as_array()
    assert output == 1
