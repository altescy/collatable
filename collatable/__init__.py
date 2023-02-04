from importlib.metadata import version

from collatable.collator import Collator, collate  # noqa: F401
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
from collatable.instance import Instance  # noqa: F401

__version__ = version("collatable")
__all__ = [
    "AdjacencyField",
    "Collator",
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
