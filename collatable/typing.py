from typing import Any, Dict, List, TypeVar, Union

from numpy.typing import ArrayLike, NDArray  # noqa: F401

Scalar = Union[int, float, complex, bool]
Tensor = NDArray
DataArray = Union[Tensor, Dict[str, Tensor], Dict[str, Any], List[Any]]
T_Scalar = TypeVar("T_Scalar", bound=Scalar)
T_Tensor = TypeVar("T_Tensor", bound=Tensor)
T_DataArray = TypeVar("T_DataArray", Tensor, Dict[str, Tensor], Dict[str, Any], List[Any])
