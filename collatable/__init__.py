from importlib.metadata import version

from collatable.fields import (  # noqa: F401
    AdjacencyField,
    Field,
    IndexField,
    LabelField,
    ListField,
    MetadataField,
    ScalarField,
    SequenceField,
    SequenceLabelField,
    SpanField,
    TensorField,
    TextField,
)
from collatable.instance import Instance, collate  # noqa: F401

__version__ = version("collatable")
__all__ = [
    "AdjacencyField",
    "Field",
    "IndexField",
    "LabelField",
    "ListField",
    "ScalarField",
    "SequenceField",
    "SpanField",
    "TensorField",
    "TextField",
    "Instance",
    "collate",
]
