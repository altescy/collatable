from typing import Any, Mapping, Sequence, TypeVar, Union

import numpy
from numpy.typing import ArrayLike, NDArray  # noqa: F401

Scalar = Union[int, float, complex, bool]
Tensor = NDArray
BoolTensor = NDArray[numpy.bool_]
IntTensor = NDArray[numpy.int_]
FloatTensor = NDArray[numpy.float32]
DataArray = Union[Tensor, Mapping[str, Tensor], Mapping[str, Any], Sequence[Any]]
ScalarT = TypeVar("ScalarT", bound=Scalar)
TensorT = TypeVar("TensorT", bound=Tensor)
DataArrayT = TypeVar("DataArrayT", bound=DataArray)
ScalarT_co = TypeVar("ScalarT_co", bound=Scalar, covariant=True)
TensorT_co = TypeVar("TensorT_co", bound=Tensor, covariant=True)
DataArrayT_co = TypeVar("DataArrayT_co", bound=DataArray, covariant=True)
