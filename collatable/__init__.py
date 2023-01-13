from importlib.metadata import version

from collatable.fields import (  # noqa: F401
    Field,
    IndexField,
    LabelField,
    ListField,
    ScalarField,
    SequenceField,
    TensorField,
    TextField,
)
from collatable.instance import Instance, collate  # noqa: F401

__version__ = version("collatable")
__all__ = [
    "Field",
    "IndexField",
    "LabelField",
    "ListField",
    "ScalarField",
    "SequenceField",
    "TensorField",
    "TextField",
    "Instance",
    "collate",
]
