import pytest

from collatable.extras.indexer import TokenIndexer


def test_token_indexer() -> None:
    tokens = list("abcde")
    indexer = TokenIndexer()
    with indexer.set(train=True):
        for token in tokens:
            indexer[token]

    assert len(indexer) == 5
    assert indexer["a"] == 0

    with pytest.raises(KeyError):
        indexer["f"]

    array = indexer(tokens)
    assert isinstance(array, dict)
    assert array["token_ids"].tolist() == [0, 1, 2, 3, 4]
    assert array["lengths"] == 5
