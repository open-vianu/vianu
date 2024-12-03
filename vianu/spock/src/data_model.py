from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
import dacite
from hashlib import sha256
import json
import logging
import os
from pathlib import Path

import numpy as np
from typing import Tuple, List, Self


@dataclass
class DataUnit(ABC):
    """Abstract base class for all data entities.

    Notes
        The identifier :param:`DataUnit.id_` is hashed and enriched with `_id_prefix` if this is
        not present. This means as long as the `id_` begins with `_id_prefix` nothing is done.

        This behavior should make it easy for to user to call
            Document(id_='This is my string that I want to hash')
        or
            Document(id_='doc_Ssdaf98safd85asfd57asdf8asdf98asdf5')
    """

    id_: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataUnit):
            return NotImplemented
        return self.id_ == other.id_

    def __post_init__(self):
        if not self.id_.startswith(self._id_prefix):
            self.id_ = self._id_prefix + self._hash_id_str()

    def _hash_id_str(self):
        return sha256(self.id_.encode()).hexdigest()

    @property
    @abstractmethod
    def _id_prefix(self):
        pass


@dataclass(eq=False)
class NamedEntity(DataUnit):
    """Class for all named entities."""
    text: str = field(default_factory=str)
    class_: str = field(default_factory=str)
    location: Tuple[int] | None = None
    probability: float | None = None

    @property
    def _id_prefix(self):
        return 'ent_'


@dataclass(eq=False)
class TextEntity(DataUnit):
    """Class containing any text-entity related information."""

    # Text information
    text: str = field(default_factory=str)
    location: List[int] = field(default_factory=list)

    # NER information (optional)
    medicinal_products: List[NamedEntity] = field(default_factory=list)
    adverse_reactions: List[NamedEntity] = field(default_factory=list)


    # private document fields
    __raw_text: str = ""

    @property
    def _id_prefix(self):
        return 'txt_'

    def add_raw_text(self, text: str) -> None:
        self.__raw_text = text

    def get_raw_text(self) -> str:
        return self.__raw_text


DOCUMENT_SOURCES = [
    'pubmed',
    'faers',
]


@dataclass(eq=False)
class Document(DataUnit):
    """Class containing any document related information."""

    url: str | None = None
    source_url: str | None = None
    source: str | None = None
    type: str | None = None
    language: str | None = None

    # optional document fields
    text: str | None = None
    title: str | None = None
    abstract: str | None = None
    publication_date: datetime | None = None

    # protected document fields (considered by `DataUnit.asdict()`)
    _text_entities: List[TextEntity] = field(default_factory=list)
    _errors: List[str] = field(default_factory=list)

    # private document fields
    __raw_text: str | None = None
    __text_entity_id_index_map: dict = field(default_factory=dict)

    @property
    def _id_prefix(self):
        return 'doc_'

    @property
    def text_entities(self):
        return self._text_entities

    def add_text_entity(self, text_entity: TextEntity) -> None:
        """Appends a text entity and updates the ID - Index map in self.text_entity_id_index_map."""
        if (id_ := text_entity.id_) in self.__text_entity_id_index_map:
            index = self.__text_entity_id_index_map[id_]
            logging.debug(f'A text entity object with id={id_} already exists at loc={index} and is ignored')
        else:
            self._text_entities.append(text_entity)
            self.__text_entity_id_index_map[text_entity.id_] = len(self.text_entities) - 1

    def del_text_entity(self, index_: int | None = None, id_: str | None = None) -> None:
        """Removes a given text entity by its index or its id."""
        if index_ is not None and id_ is not None:
            raise ValueError('The definition of `index` OR `id_` must be exclusive.')
        if index_ is None and id_ is None:
            raise ValueError('Either `index` or `id_` has to be defined.')
        if id_ is not None:
            index = self.__text_entity_id_index_map[id_]
        elif index_ is not None:
            index = index_
        self._text_entities.pop(index)
        self.__text_entity_id_index_map = {ent.id_: i for i, ent in enumerate(self._text_entities)}

    def add_error(self, err: str) -> None:
        self._errors.append(err)

    def get_errors(self) -> List[str]:
        return self._errors

    def has_error(self) -> bool:
        return len(self._errors) > 0

    def add_raw_text(self, text: str) -> None:
        self.__raw_text = text

    def get_raw_text(self) -> str:
        return self.__raw_text
    
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(type_hooks={datetime: datetime.fromisoformat})
        )


class DocumentJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Document):
            return asdict(o)
        if isinstance(o, np.float32):
            return str(o)
        if isinstance(o, np.int64):
            return int(o)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)



class FileHandler:
    """Reads from and write data to a file.
    """
    _suffix = '.json'

    def __init__(self, data_file: Path | str):
        data_file = Path(data_file) if isinstance(data_file, str) else data_file
        self._data_file = data_file.with_suffix(self._suffix)
        if not self._data_file.parent.exists():
            os.makedirs(self._data_file.parent)
        logging.debug(f"FileHandler path: '{self._data_file}'")


    def read(self) -> List[Document]:
        with open(self._data_file, 'r', encoding="utf-8") as dfile:
            entries = json.load(dfile)
        data = [Document.from_dict(x) for x in entries]
        return data

    def write(self, data: List[Document]):
        with open(self._data_file, 'w', encoding="utf-8") as dfile:
            json.dump(data, dfile, cls=DocumentJSONEncoder)
