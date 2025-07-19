from typing import List, Mapping, Sequence, Type, cast

import numpy

from collatable.types import ArrayLike, DataArray, ScalarT, TensorT


def stack_with_padding(
    tensors: Sequence[TensorT],
    padding_value: ArrayLike = 0,
) -> TensorT:
    num_arrays = len(tensors)
    shapes = tuple(tensor.shape for tensor in tensors)
    max_shape = tuple(max(shape) for shape in zip(*shapes))
    if max_shape == ():
        return cast(TensorT, numpy.array(tensors))
    stacked = cast(
        TensorT, numpy.zeros((num_arrays, *max_shape), dtype=tensors[0].dtype)
    )
    for index, tensor in enumerate(tensors):
        stacked[index] = numpy.pad(
            tensor,
            [(0, max_shape[dim] - tensor.shape[dim]) for dim in range(len(max_shape))],
            constant_values=padding_value,
        )
    return stacked


def get_scalar_default_value(cls: Type[ScalarT]) -> ScalarT:
    if issubclass(cls, bool):
        return cast(ScalarT, False)
    if issubclass(cls, int):
        return cast(ScalarT, 0)
    if issubclass(cls, float):
        return cast(ScalarT, 0.0)
    if issubclass(cls, complex):
        return cast(ScalarT, 0.0 + 0.0j)
    raise TypeError(f"Unsupported type: {cls}")


def debatched(array: DataArray) -> List[DataArray]:
    if isinstance(array, (Sequence, numpy.ndarray)):
        return list(array)
    if isinstance(array, Mapping):
        keys = set(array)
        debatched_values = {key: debatched(value) for key, value in array.items()}
        batch_size = len(debatched_values[next(iter(keys))])
        return [
            {key: debatched_values[key][index] for key in keys}
            for index in range(batch_size)
        ]
    raise TypeError(f"Unsupported type: {type(array)}")
