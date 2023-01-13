from typing import Sequence, Type, cast

import numpy

from collatable.typing import ArrayLike, T_Scalar, T_Tensor


def stack_with_padding(
    tensors: Sequence[T_Tensor],
    padding_value: ArrayLike = 0,
) -> T_Tensor:
    num_arrays = len(tensors)
    shapes = tuple(tensor.shape for tensor in tensors)
    max_shape = tuple(max(shape) for shape in zip(*shapes))
    if max_shape == ():
        return cast(T_Tensor, numpy.array(tensors))
    stacked = cast(T_Tensor, numpy.zeros((num_arrays, *max_shape), dtype=tensors[0].dtype))
    for index, tensor in enumerate(tensors):
        stacked[index] = numpy.pad(
            tensor,
            [(0, max_shape[dim] - tensor.shape[dim]) for dim in range(len(max_shape))],
            constant_values=padding_value,
        )
    return stacked


def get_scalar_default_value(cls: Type[T_Scalar]) -> T_Scalar:
    if issubclass(cls, int):
        return cast(T_Scalar, 0)
    if issubclass(cls, float):
        return cast(T_Scalar, 0.0)
    if issubclass(cls, complex):
        return cast(T_Scalar, 0.0 + 0.0j)
    if issubclass(cls, bool):
        return cast(T_Scalar, False)
    raise TypeError(f"Unsupported type: {cls}")
