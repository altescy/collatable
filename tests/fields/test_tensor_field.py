import numpy

from collatable.fields.tensor_field import TensorField


def test_tensor_field() -> None:
    fields = [
        TensorField(numpy.zeros((2, 3)), padding_value=-1),
        TensorField(numpy.ones((3, 4)), padding_value=-1),
    ]
    output = fields[0].collate(fields)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 3, 4)  # type: ignore[comparison-overlap]
    assert output.sum() == 6
    numpy.testing.assert_array_equal(output[0, :2, :3], numpy.zeros((2, 3)))
    numpy.testing.assert_array_equal(output[1, :3, :4], numpy.ones((3, 4)))
