import numpy

from collatable.fields.scalar_field import ScalarField


def test_scalar_field() -> None:
    fields = [ScalarField(value) for value in range(5)]
    output = fields[0].collate(fields)
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (5,)  # type: ignore[comparison-overlap]
    assert output.tolist() == [0, 1, 2, 3, 4]
