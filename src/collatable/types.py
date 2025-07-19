import dataclasses
from typing import (
    Any,
    ClassVar,
    Dict,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ArrayLike",
    "BoolTensor",
    "DataArray",
    "DataArrayT",
    "DataArrayT_co",
    "DataclassT",
    "IDataclass",
    "IntTensor",
    "INamedTuple",
    "NamedTupleT",
    "Scalar",
    "ScalarT",
    "ScalarT_co",
    "Tensor",
    "TensorT",
    "TensorT_co",
]

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


@runtime_checkable
class IDataclass(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, dataclasses.Field]]


@runtime_checkable
class INamedTuple(Protocol):
    _fields: ClassVar[Tuple[str, ...]]

    def _asdict(self) -> Dict[str, Any]: ...

    def _replace(self: "NamedTupleT", **kwargs: Any) -> "NamedTupleT": ...


DataclassT = TypeVar("DataclassT", bound=IDataclass)
NamedTupleT = TypeVar("NamedTupleT", bound=INamedTuple)
