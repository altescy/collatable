import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    runtime_checkable,
)

from collatable import Field, LabelField, TextField
from collatable.typing import DataArray, Scalar, Tensor

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
HashableT = TypeVar("HashableT", bound=Hashable)
HashableT_contra = TypeVar("HashableT_contra", bound=Hashable, contravariant=True)
IndexT_co = TypeVar("IndexT_co", bound=Union[Scalar, DataArray], covariant=True)


@runtime_checkable
class IIndexer(Protocol[HashableT_contra, IndexT_co]):
    def __len__(self) -> int: ...

    def __getitem__(self, value: HashableT_contra, /) -> int: ...

    def __call__(self, value: HashableT_contra, /) -> IndexT_co: ...


@runtime_checkable
class ISequenceIndexer(Protocol[HashableT_contra, IndexT_co]):
    def __len__(self) -> int: ...

    def __getitem__(self, value: HashableT_contra, /) -> int: ...

    def __call__(self, values: Sequence[HashableT_contra], /) -> IndexT_co: ...


class FieldAccessor:
    def __init__(self, field: str) -> None:
        self._field = field.split(".")

    def __call__(self, obj: Any) -> Any:
        for part in self._field:
            obj = obj[part] if isinstance(obj, Mapping) else getattr(obj, part)
        return obj


class FieldTransform(Generic[S]):
    def __call__(self, obj: S) -> Field:
        raise NotImplementedError

    def build(self, dataset: Iterable[S]) -> None:
        pass

    def indexers(self) -> Mapping[str, IIndexer]:
        return dict((attribute, value) for attribute, value in self.__dict__.items() if isinstance(value, IIndexer))


class TextFieldTransform(Generic[HashableT], FieldTransform[Union[str, Sequence[HashableT]]]):
    _DEFAULT_TOKENIZER_PATTERN = re.compile(r"(?u)\b\w\w+\b")

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], Sequence[HashableT]]] = None,
        pad_token: Optional[HashableT] = None,
        unk_token: Optional[HashableT] = None,
        special_tokens: Optional[Sequence[HashableT]] = None,
        indexer: Optional[ISequenceIndexer[HashableT, Mapping[str, Tensor]]] = None,
    ) -> None:
        from .indexer import TokenIndexer

        self._tokenizer = tokenizer or (lambda text: self._DEFAULT_TOKENIZER_PATTERN.findall(text))
        self._indexer: ISequenceIndexer[HashableT, Mapping[str, Tensor]] = (
            indexer if indexer is not None else TokenIndexer[HashableT]()
        )
        self._pad_token = pad_token
        self._special_tokens = special_tokens or []
        if unk_token is not None and unk_token not in self._special_tokens:
            self._special_tokens = [unk_token, *self._special_tokens]
        if pad_token is not None and pad_token not in self._special_tokens:
            self._special_tokens = [pad_token, *self._special_tokens]

    def __call__(self, obj: Union[str, Sequence[HashableT]]) -> TextField:
        if isinstance(obj, str):
            obj = self._tokenizer(obj)
        return TextField(
            obj,
            indexer=self._indexer,
            padding_value=self._indexer[self._pad_token] if self._pad_token is not None else 0,
        )

    def build(self, dataset: Iterable[Union[str, Sequence[HashableT]]]) -> None:
        for special_token in self._special_tokens:
            self._indexer[special_token]
        for text in dataset:
            if isinstance(text, str):
                text = self._tokenizer(text)
            self._indexer(text)

    def indexers(self) -> Mapping[str, IIndexer]:
        return {"tokens": self._indexer}


class LabelFieldTransform(FieldTransform[HashableT]):
    def __init__(
        self,
        indexer: Optional[IIndexer[HashableT, int]] = None,
    ) -> None:
        from .indexer import LabelIndexer

        self._indexer: IIndexer[HashableT, int] = indexer or LabelIndexer[HashableT]()

    def __call__(self, obj: HashableT) -> LabelField:
        return LabelField(obj, indexer=self._indexer)

    def build(self, dataset: Iterable[HashableT]) -> None:
        for label in dataset:
            self._indexer[label]

    def indexers(self) -> Mapping[str, IIndexer[HashableT, int]]:
        return {"labels": self._indexer}


@dataclass
class FieldConfig(Generic[S, T]):
    accessor: Callable[[S], T]
    transform: FieldTransform[T]


class DataModule(Generic[T]):
    def __init__(
        self,
        fields: Mapping[str, Union[FieldTransform, FieldConfig[T, Any]]],
    ) -> None:
        self._fields: Mapping[str, FieldConfig[T, Any]] = {
            name: (
                FieldConfig(accessor=FieldAccessor(name), transform=transform)
                if isinstance(transform, FieldTransform)
                else transform
            )
            for name, transform in fields.items()
        }

    def build(self, dataset: Iterable[T]) -> None:
        for field in self._fields.values():
            field.transform.build(field.accessor(obj) for obj in dataset)

    def __call__(self, dataset: Iterable[T]) -> Iterable[Dict[str, Field]]:
        for obj in dataset:
            yield {name: field.transform(field.accessor(obj)) for name, field in self._fields.items()}

    def indexer(self, field: str, name: str) -> IIndexer:
        return self._fields[field].transform.indexers()[name]
