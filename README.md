# Collatable

[![Actions Status](https://github.com/altescy/collatable/workflows/CI/badge.svg)](https://github.com/altescy/collatable/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/collatable)](https://github.com/altescy/collatable/blob/main/LICENSE)
[![Python version](https://img.shields.io/pypi/pyversions/collatable)](https://github.com/altescy/collatable)
[![pypi version](https://img.shields.io/pypi/v/collatable)](https://pypi.org/project/collatable/)

Constructing batched tensors for any machine learning tasks

## Installation

```bash
pip install collatable
```

## Examples

The following scripts show how to tokenize/index/collate your dataset with `collatable`:

### Text Classification

```python
import collatable
from collatable import Instance, LabelField, MetadataField, TextField
from collatable.extras.indexer import LabelIndexer, TokenIndexer

dataset = [
    ("this is awesome", "positive"),
    ("this is a bad movie", "negative"),
    ("this movie is an awesome movie", "positive"),
    ("this movie is too bad to watch", "negative"),
]

# Set up indexers for tokens and labels
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
token_indexer = TokenIndexer[str](specials=[PAD_TOKEN, UNK_TOKEN], default=UNK_TOKEN)
label_indexer = LabelIndexer[str]()

# Load training dataset
instances = []
with token_indexer.context(train=True), label_indexer.context(train=True):
    for id_, (text, label) in enumerate(dataset):
        # Prepare each field with the corresponding field class
        text_field = TextField(
            text.split(),
            indexer=token_indexer,
            padding_value=token_indexer[PAD_TOKEN],
        )
        label_field = LabelField(
            label,
            indexer=label_indexer,
        )
        metadata_field = MetadataField({"id": id_})
        # Combine these fields into instance
        instance = Instance(
            text=text_field,
            label=label_field,
            metadata=metadata_field,
        )
        instances.append(instance)

# Collate instances and build batch
output = collatable.collate(instances)
print(output)
```

Execution result:

```text
{'metadata': [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}],
 'text': {
    'token_ids': array([[ 2,  3,  4,  0,  0,  0,  0],
                        [ 2,  3,  5,  6,  7,  0,  0],
                        [ 2,  7,  3,  8,  4,  7,  0],
                        [ 2,  7,  3,  9,  6, 10, 11]]),
    'mask': array([[ True,  True,  True, False, False, False, False],
                   [ True,  True,  True,  True,  True, False, False],
                   [ True,  True,  True,  True,  True,  True, False],
                   [ True,  True,  True,  True,  True,  True,  True]])},
 'label': array([0, 1, 0, 1], dtype=int32)}
```

### Sequence Labeling

```python
import collatable
from collatable import Instance, SequenceLabelField, TextField
from collatable.extras.indexer import LabelIndexer, TokenIndexer

dataset = [
    (["my", "name", "is", "john", "smith"], ["O", "O", "O", "B", "I"]),
    (["i", "lived", "in", "japan", "three", "years", "ago"], ["O", "O", "O", "U", "O", "O", "O"]),
]

# Set up indexers for tokens and labels
PAD_TOKEN = "<PAD>"
token_indexer = TokenIndexer[str](specials=(PAD_TOKEN,))
label_indexer = LabelIndexer[str]()

# Load training dataset
instances = []
with token_indexer.context(train=True), label_indexer.context(train=True):
    for tokens, labels in dataset:
        text_field = TextField(tokens, indexer=token_indexer, padding_value=token_indexer[PAD_TOKEN])
        label_field = SequenceLabelField(labels, text_field, indexer=label_indexer)
        instance = Instance(text=text_field, label=label_field)
        instances.append(instance)

output = collatable.collate(instances)
print(output)
```

Execution result:

```text
{'label': array([[0, 0, 0, 1, 2, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0]]),
 'text': {
    'token_ids': array([[ 1,  2,  3,  4,  5,  0,  0],
                        [ 6,  7,  8,  9, 10, 11, 12]]),
    'mask': array([[ True,  True,  True,  True,  True, False, False],
                   [ True,  True,  True,  True,  True,  True,  True]])}}
```

### Relation Extraction

```python
import collatable
from collatable.extras.indexer import LabelIndexer, TokenIndexer
from collatable import AdjacencyField, Instance, ListField, SpanField, TextField

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

output = collatable.collate(instances)
print(output)
```

Execution result:

```text
{'text': {
    'token_ids': array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  5, 11],
                        [11, 12, 13, 14, 15, 16,  0,  0,  0,  0,  0,  0]]),
    'mask': array([[ True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True],
                   [ True,  True,  True,  True,  True,  True, False, False, False, False, False, False]])},
 'spans': array([[[ 0,  2],
                  [ 5,  7],
                  [11, 12]],
                 [[ 0,  1],
                  [ 5,  6],
                  [-1, -1]]]),
 'relations': array([[[-1,  0,  1],
                      [-1, -1, -1],
                      [-1, -1, -1]],
                     [[-1,  2, -1],
                      [-1, -1, -1],
                      [-1, -1, -1]]], dtype=int32)}
```
