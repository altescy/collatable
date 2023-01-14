# Collatable

[![Actions Status](https://github.com/altescy/collatable/workflows/CI/badge.svg)](https://github.com/altescy/collatable/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/collatable)](https://github.com/altescy/collatable/blob/main/LICENSE)
[![Python version](https://img.shields.io/pypi/pyversions/collatable)](https://github.com/altescy/collatable)
[![pypi version](https://img.shields.io/pypi/v/collatable)](https://pypi.org/project/collatable/)

Constructing batched tensors for any machine learning tasks

## Examples

The following scripts show how to tokenize/index/collate your dataset with `collatable`:

### Text Classification

```python
from typing import List

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
instances: List[Instance] = []
with token_indexer.context(train=True), label_indexer.context(train=True):
    for id_, (text, label) in dataset:
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
        metadata_field = Metadata({"id": id_})
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
{'label': array([0, 1, 0, 1], dtype=int32),
 'metadata': [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}],
 'text': {
    'token_ids': array([[ 3,  4,  5,  0,  0,  0,  0],
                        [ 3,  4,  6,  7,  8,  0,  0],
                        [ 3,  8,  4,  9,  5,  8,  0],
                        [ 3,  8,  4, 10,  7, 11, 12]]),
    'lengths': array([3, 5, 6, 7])}}
```

### Sequence Labeling

```python
from typing import List

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
instances: List[Instance] = []
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
{'text': {
  'token_ids': array([[ 1,  2,  3,  4,  5,  0,  0],
                     [ 6,  7,  8,  9, 10, 11, 12]]),
  'lengths': array([5, 7])},
 'label': array([[0, 0, 0, 1, 2, 0, 0],
                 [0, 0, 0, 3, 0, 0, 0]])}
```
