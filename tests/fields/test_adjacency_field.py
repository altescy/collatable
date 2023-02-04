import numpy

from collatable.collator import collate
from collatable.extras.indexer import LabelIndexer, TokenIndexer
from collatable.fields.adjacency_field import AdjacencyField
from collatable.fields.list_field import ListField
from collatable.fields.span_field import SpanField
from collatable.fields.text_field import TextField
from collatable.instance import Instance


def test_adajacency_field() -> None:
    PAD_TOKEN = "<PAD>"
    token_indexer = TokenIndexer[str](specials=(PAD_TOKEN,))
    label_indexer = LabelIndexer[str]()

    instances = []
    with token_indexer.context(train=True), label_indexer.context(train=True):
        text = TextField(
            ["john", "smith", "was", "born", "in", "new", "york", "and", "now", "lives", "in", "tokyo"],
            indexer=token_indexer,
            padding_value=token_indexer[PAD_TOKEN],
        )
        spans = ListField([SpanField(0, 2, text), SpanField(5, 7, text), SpanField(11, 12, text)])
        relations = AdjacencyField([(0, 1), (0, 2)], spans, labels=["born-in", "lives-in"], indexer=label_indexer)
        instance = Instance(text=text, spans=spans, relations=relations)
        instances.append(instance)

        text = TextField(
            ["tokyo", "is", "the", "capital", "of", "japan"],
            indexer=token_indexer,
            padding_value=token_indexer[PAD_TOKEN],
        )
        spans = ListField([SpanField(0, 1, text), SpanField(5, 6, text)])
        relations = AdjacencyField([(0, 1)], spans, labels=["capital-of"], indexer=label_indexer)
        instance = Instance(text=text, spans=spans, relations=relations)
        instances.append(instance)

    output = collate(instances)["relations"]
    assert isinstance(output, numpy.ndarray)
    assert output.shape == (2, 3, 3)

    relation_counts = (output >= 0).sum(2).sum(1)
    assert relation_counts.tolist() == [2, 1]
