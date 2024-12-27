from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, asdict, field
from datetime import datetime
from hashlib import sha256
import json
import logging
import os
from pathlib import Path

import dacite
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

    # additional document fields
    title: str | None = None
    url: str | None = None
    source_url: str | None = None
    source_favicon_url: str | None = None
    language: str | None = None
    publication_date: datetime | None = None

    # named entities
    medicinal_products: List[NamedEntity] = field(default_factory=list)
    adverse_reactions: List[NamedEntity] = field(default_factory=list)

    # protected fields
    _html: str | None = None
    _html_hash: str | None = None

    @property
    def _id_prefix(self):
        return 'doc_'
    
    def remove_named_entity_by_id(self, id_: str) -> None:
        """Removes a named entity from the document by a given `doc.id_`."""
        self.medicinal_products = [ne for ne in self.medicinal_products if ne.id_ != id_]
        self.adverse_reactions = [ne for ne in self.adverse_reactions if ne.id_ != id_]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Creates a :class:`Document` object from a dictionary."""
        return dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(type_hooks={datetime: datetime.fromisoformat})
        )
    
    def _get_html_hash(self) -> str:
        """Creates a sha256 hash from the named entities' ids. If the sets of named entities have been modified, this
        function will return a different hash.
        """
        ne_ids = [ne.id_ for ne in self.medicinal_products + self.adverse_reactions]
        html_hash_str = ' '.join(ne_ids)
        return sha256(html_hash_str.encode()).hexdigest()


    def _get_html(self) -> str:
        """Creates the HTML representation of the document with highlighted named entities."""
        text = f"<div>{self.text}</div>"

        # Highlight medicinal products accodring to the css class 'mp'
        mp_template = "<span class='ner mp'>{text} | {class_}</span>"
        for ne in self.medicinal_products:
            text = text.replace(
                ne.text, mp_template.format(text=ne.text, class_=ne.class_)
            )
        
        # Highlight adverse drug reactions accodring to the css class 'adr'
        adr_template = "<span class='ner adr'>{text} | {class_}</span>"
        for ne in self.adverse_reactions:
            text = text.replace(
                ne.text, adr_template.format(text=ne.text, class_=ne.class_)
            )
        
        return text


    def get_html(self) -> str:
        """Returns the HTML representation of the document with highlighted named entities. This function checks if 
        the set of named entities has been modified and updates the HTML representation if necessary."""
        html_hash = self._get_html_hash()
        if self._html is None or html_hash != self._html_hash:
            self._html = self._get_html()
            self._html_hash = html_hash
        return self._html


class DocumentJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for the :class:`Document` class."""
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
    """Class for the job definition resulting from the cli arguments."""

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

    def __post_init__(self) -> None:
        super().__post_init__()
        self.submission = datetime.now() if self.submission is None else self.submission

    @property
    def _id_prefix(self) -> str:
        return 'job_'
    
    def to_namespace(self) -> Namespace:
        """Converts the :class:`Job` object to a :class:`argparse.Namespace` object."""
        return Namespace(**asdict(self))
    

@dataclass
class QueueItem:
    """Class for the :class:`asyncio.Queue` items"""
    id_: str
    doc: Document


@dataclass
class SpoCK:
    """Main class for the SpoCK pipeline mainly containing the job definition and the resulting data."""
    # Generic fields
    status: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Pipeline fields
    job: Job | None = None
    data: List[Document] = field(default_factory=list)

    def runtime(self) -> datetime | None:
        if self.started_at is not None:
            if self.finished_at is None:
                return datetime.now() - self.started_at
            return self.finished_at - self.started_at
        return None
    

class FileHandler:
    """Reads from and write data to a JSON file under a given file path."""

    _suffix = '.json'

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path) if isinstance(path, str) else path
        if not self._path.exists():
            os.makedirs(self._path)


    def read(self, file: str) -> List[Document]:
        """Reads the data from a JSON file and casts it into a list of :class:`Document` objects."""
        filename = (self._path / file).with_suffix(self._suffix)

        logger.info('reading data from file {filename}')
        with open(filename.with_suffix(self._suffix), 'r', encoding="utf-8") as dfile:
            entries = json.load(dfile)
        
        return [Document.from_dict(x) for x in entries]


    def write(self, file: str, data: List[Document]) -> None:
        """Writes the data to a JSON file with name `[file]_%Y%m%d%H%M%S`."""
        file = f'{file}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        filename = (self._path / file).with_suffix(self._suffix)

        logger.info(f'writing data to file {filename}')
        with open(filename, 'w', encoding="utf-8") as dfile:
            json.dump(data, dfile, cls=DocumentJSONEncoder)
