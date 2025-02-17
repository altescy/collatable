import numpy

from collatable.fields.list_field import ListField
from collatable.fields.scalar_field import ScalarField
from collatable.fields.tensor_field import TensorField


def test_list_field_can_convert_scalara_fields_to_array() -> None:
    field = ListField([ScalarField(value) for value in range(5)])
    output = field.as_array()
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (5,)  # type: ignore[comparison-overlap]


def test_list_field_can_convert_tensor_fields_to_array() -> None:
    field = ListField(
        [
            TensorField(numpy.zeros((2, 3)), padding_value=-1),
            TensorField(numpy.ones((3, 4)), padding_value=-1),
        ]
    )
    output = field.as_array()
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 3, 4)  # type: ignore[comparison-overlap]
    numpy.testing.assert_array_equal(output[0, :2, :3], numpy.zeros((2, 3)))
    numpy.testing.assert_array_equal(output[1, :3, :4], numpy.ones((3, 4)))


def test_list_field_can_convert_nested_list_fields_to_array() -> None:
    field = ListField(
        [
            ListField([ScalarField(value) for value in range(2)]),
            ListField([ScalarField(value) for value in range(3)]),
        ]
    )
    output = field.as_array()
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 3)
    assert output[0].tolist() == [0, 1, 0]
    assert output[1].tolist() == [0, 1, 2]


def test_list_field_with_scalar_fields() -> None:
    fields = [
        ListField([ScalarField(value) for value in range(3)]),
        ListField([ScalarField(value) for value in range(5)]),
    ]
    output = fields[0].collate(fields)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 5)
    assert output[0].tolist() == [0, 1, 2, 0, 0]
    assert output[1].tolist() == [0, 1, 2, 3, 4]
