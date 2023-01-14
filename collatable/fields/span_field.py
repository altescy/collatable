import numpy

from collatable.fields.field import Field, PaddingValue
from collatable.fields.sequence_field import SequenceField
from collatable.typing import Tensor


class SpanField(Field[Tensor]):
    __slots__ = ["span_start", "span_end", "sequence_field", "padding_value"]

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

        self.span_start = span_start
        self.span_end = span_end
        self.sequence_field = sequence_field

    def as_array(self) -> Tensor:
        return numpy.array([self.span_start, self.span_end])
