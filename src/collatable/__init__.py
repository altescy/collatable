from importlib.metadata import version

from collatable.collator import Collator, collate
from collatable.fields import (
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

__version__ = version("collatable")
__all__ = [
    "AdjacencyField",
    "Collator",
    "Field",
    "IndexField",
    "LabelField",
    "ListField",
    "MetadataField",
    "ScalarField",
    "SequenceField",
    "SequenceLabelField",
    "SpanField",
    "TensorField",
    "TextField",
    "collate",
]
