from abc import ABC, abstractmethod
from argparse import Namespace
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


logger = logging.getLogger(__name__)


@dataclass
class DataUnit(ABC):
    """Abstract base class for all dataclass entities with customized id.

    Notes
        The identifier :param:`DataUnit.id_` is hashed and enriched with `_id_prefix` if this is
        not present. This means as long as the `id_` begins with `_id_prefix` nothing is done.

        This behavior aims to allow:
            SubDataUnit(id_='This is the string that identiies the entity')
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
    source_favicon_url: str | None = None
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


@dataclass(eq=False)
class Job(DataUnit):

    # generic options
    log_level: str

    # scraping options
    term: str
    source: List[str]

    # NER options
    model: str
    n_ner_tasks: int

    # optional fields
    submission: datetime | None = None
    data_path: str | None = None
    data_file: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.submission = datetime.now() if self.submission is None else self.submission

    @property
    def _id_prefix(self):
        return 'job_'
    
    def to_namespace(self):
        return Namespace(**asdict(self))
    

@dataclass
class QueueItem:
    """Class for the async queue items"""
    id_: str
    doc: Document


@dataclass
class SpoCK:
    # Generic fields
    status: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Pipeline fields
    job: Job | None = None
    data: List[Document] | None = None

    def runtime(self) -> datetime | None:
        if self.started_at is not None:
            if self.completed_at is None:
                return datetime.now() - self.started_at
            return self.completed_at - self.started_at
        return None
    

class FileHandler:
    """Reads from and write data to a file."""

    _suffix = '.json'

    def __init__(self, path: Path | str):
        self._path = Path(path) if isinstance(path, str) else path
        if not self._path.exists():
            os.makedirs(self._path)


    def read(self, file: str) -> List[Document]:
        filename = (self._path / file).with_suffix(self._suffix)
        logger.info('reading data from file {filename}')
        with open(filename.with_suffix(self._suffix), 'r', encoding="utf-8") as dfile:
            entries = json.load(dfile)
        data = [Document.from_dict(x) for x in entries]
        return data


    def write(self, file: str, data: List[Document]):
        file = f'{file}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        filename = (self._path / file).with_suffix(self._suffix)
        logger.info(f'writing data to file {filename}')
        with open(filename, 'w', encoding="utf-8") as dfile:
            json.dump(data, dfile, cls=DocumentJSONEncoder)
