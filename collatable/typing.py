from typing import Any, Dict, List, TypeVar, Union

from numpy.typing import ArrayLike, NDArray  # noqa: F401

Scalar = Union[int, float, complex, bool]
Tensor = NDArray
DataArray = Union[Tensor, Dict[str, Tensor], Dict[str, Any], List[Any]]
ScalarT = TypeVar("ScalarT", bound=Scalar)
TensorT = TypeVar("TensorT", bound=Tensor)
DataArrayT = TypeVar("DataArrayT", Tensor, Dict[str, Tensor], Dict[str, Any], List[Any])
