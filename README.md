# Collatable

[![Actions Status](https://github.com/altescy/collatable/workflows/CI/badge.svg)](https://github.com/altescy/collatable/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/collatable)](https://github.com/altescy/collatable/blob/main/LICENSE)


## Usage

### Example of text classification

This is an example of building bached tensors with text classification dataset.
The following script shows how to tokenize/index/collate your dataset with `collatable`:

```python
from typing import List

import collatable
from collatable import Instance, LabelField, TextField
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
token_indexer = TokenIndexer(spetials=[PAD_TOKEN, UNK_TOKEN], default=UNK_TOKEN)
label_indexer = LabelIndexer()

instances: List[Instance] = []

# Load training dataset
with token_indexer.set(train=True), label_indexer.set(train=True):
    for text, label in dataset:
        text_field = TextField(
            text.split(),
            indexer=token_indexer,
            padding_value=token_indexer["<pad>"],
        )
        label_field = LabelField(
            label,
            indexer=label_indexer,
        )
        instance = Instance(text_field=text_field, label_field=label_field)
        instances.append(instance)

    # Collate instances: note that this should be done inside
    # the with clause because indexing is conducted here.
    output = collatable.collate(instances)

print(output)
```

Execution result:

```text
{'label_field': array([0, 1, 0, 1], dtype=int32),
 'text_field': {
    'token_ids': array([[ 3,  4,  5,  0,  0,  0,  0],
                        [ 3,  4,  6,  7,  8,  0,  0],
                        [ 3,  8,  4,  9,  5,  8,  0],
                        [ 3,  8,  4, 10,  7, 11, 12]]),
    'lengths': array([3, 5, 6, 7])
 }}
```
