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
from typing import List, Self


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
    location: List[int] | None = None

    @property
    def _id_prefix(self):
        return 'ent_'


@dataclass(eq=False)
class Document(DataUnit):
    """Class containing any document related information."""

    # mandatory document fields
    text: str
    source: str

    # optional document fields
    title: str | None = None
    url: str | None = None
    source_url: str | None = None
    language: str | None = None
    publication_date: datetime | None = None

    # NER information (optional)
    medicinal_products: List[NamedEntity] = field(default_factory=list)
    adverse_reactions: List[NamedEntity] = field(default_factory=list)

    # Protected fields
    _errors: List[str] = field(default_factory=list)

    @property
    def _id_prefix(self):
        return 'doc_'
    
    def remove_named_entity_from_id(self, id_: str) -> None:
        """Removes a named entity from the text entity."""
        self.medicinal_products = [ne for ne in self.medicinal_products if ne.id_ != id_]
        self.adverse_reactions = [ne for ne in self.adverse_reactions if ne.id_ != id_]

    def add_error(self, err: str) -> None:
        self._errors.append(err)

    def get_errors(self) -> List[str]:
        return self._errors

    def has_error(self) -> bool:
        return len(self._errors) > 0

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


@dataclass
class Query:
    term: str
    sources: List[str] | None = None
    submission_date: datetime | None = None


class FileHandler:
    """Reads from and write data to a file."""

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
