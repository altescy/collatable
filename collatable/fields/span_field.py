import numpy

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.types import IntTensor


class SpanField(Field[IntTensor]):
    __slots__ = ["_span_start", "_span_end", "_padding_value"]

    def __init__(
        self,
        span_start: int,
        span_end: int,
        sequence_field: SequenceField,
        padding_value: PaddingValue = -1,
    ) -> None:
        if span_start < 0:
            raise ValueError("Span start must be non-negative.")
        if span_end < 0:
            raise ValueError("Span end must be non-negative.")
        if span_start > span_end:
            raise ValueError("Span start must be less than or equal to span end.")
        if span_end > len(sequence_field):
            raise ValueError("Span end must be less than or equal to the length of the sequence.")

        super().__init__(padding_value=padding_value)

        self._span_start = span_start
        self._span_end = span_end

    def __str__(self) -> str:
        return f"({self.span_start}, {self.span_end})"

    def __repr__(self) -> str:
        return f"SpanField(span_start={self.span_start}, span_end={self.span_end}, padding_value={self.padding_value})"

    @property
    def span_start(self) -> int:
        return self._span_start

    @property
    def span_end(self) -> int:
        return self._span_end

    def as_array(self) -> IntTensor:
        return numpy.array([self.span_start, self.span_end])

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: IntTensor,
        sequence_field: SequenceField,
    ) -> "SpanField":
        if array.ndim != 1 or array.shape[0] != 2:
            raise ValueError(f"SpanField expects a 1-dimensional array of length 2, but got shape {array.shape}")
        return cls(array[0], array[1], sequence_field)
