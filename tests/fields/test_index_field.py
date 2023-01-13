import pytest

from collatable.fields.index_field import IndexField
from collatable.fields.text_field import TextField


def test_index_field_can_be_converted_to_array() -> None:
    vocab = {"a": 0, "is": 1, "test": 2, "this": 3}
    text = TextField(["this", "is", "a", "test"], vocab=vocab)
    field = IndexField(2, text)
    output = field.as_array()
    assert output == 2


def test_index_field_can_raise_value_error() -> None:
    vocab = {"a": 0, "is": 1, "test": 2, "this": 3}
    text = TextField(["this", "is", "a", "test"], vocab=vocab)
    with pytest.raises(ValueError, match="Index 4 is out of range for sequence of length 4"):
        IndexField(4, text)
