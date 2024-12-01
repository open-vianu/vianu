from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
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
class Serializable:
    """Abstract base class for all dataclasses that can be serialized to a dictionary."""

    def to_dict(self) -> dict:
        """Converts the object to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, dict_: dict) -> Self:
        """Creates an object from a dictionary."""
        return dacite.from_dict(
            data_class=cls,
            data=dict_,
            config=dacite.Config(type_hooks={datetime: datetime.fromisoformat}),
        )


@dataclass
class Identicator(ABC):
    """Abstract base class for entities with customized id.

    Notes
        The identifier :param:`Identicator.id_` is hashed and enriched with `_id_prefix` if this is
        not present. This means as long as the `id_` begins with `_id_prefix` nothing is done.

        This behavior aims to allow:
            SubIdenticator(id_='This is the string that identifies the entity')

        and with _id_prefix='sub' it produces an id_ of the form:
            id_ = 'sub_5d41402abc4b2a76b9719d911017c592'
    """

    id_: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identicator):
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
class NamedEntity(Identicator, Serializable):
    """Class for all named entities."""

    text: str = field(default_factory=str)
    class_: str = field(default_factory=str)
    location: List[int] | None = None

    @property
    def _id_prefix(self):
        return "ent_"


@dataclass(eq=False)
class Document(Identicator, Serializable):
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
        return "doc_"

    def remove_named_entity_by_id(self, id_: str) -> None:
        """Removes a named entity from the document by a given `doc.id_`."""
        self.medicinal_products = [
            ne for ne in self.medicinal_products if ne.id_ != id_
        ]
        self.adverse_reactions = [ne for ne in self.adverse_reactions if ne.id_ != id_]

    def _get_html_hash(self) -> str:
        """Creates a sha256 hash from the named entities' ids. If the sets of named entities have been modified, this
        function will return a different hash.
        """
        ne_ids = [ne.id_ for ne in self.medicinal_products + self.adverse_reactions]
        html_hash_str = " ".join(ne_ids)
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


@dataclass(eq=False)
class Setup(Identicator, Serializable):
    """Class for the pipeline setup (closely related to the CLI arguments)."""

    # generic options
    log_level: str
    max_docs_src: int

    # scraping options
    term: str
    source: List[str]
    n_scp_tasks: int

    # NER options
    model: str
    n_ner_tasks: int

    # optional fields
    submission: datetime | None = None
    file_path: str | None = None
    file_name: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.submission = datetime.now() if self.submission is None else self.submission

    @property
    def _id_prefix(self) -> str:
        return "stp_"

    def to_namespace(self) -> Namespace:
        """Converts the :class:`Setup` object to a :class:`argparse.Namespace` object."""
        return Namespace(**asdict(self))

    @classmethod
    def from_namespace(cls, args_: Namespace) -> Self:
        """Creates a :class:`Setup` object from a :class:`argparse.Namespace` object."""
        args_dict = vars(args_)
        return cls(id_=str(args_dict), **args_dict)


@dataclass
class QueueItem:
    """Class for the :class:`asyncio.Queue` items"""

    id_: str
    doc: Document


@dataclass
class SpoCK(Identicator, Serializable):
    """Main class for the SpoCK pipeline mainly containing the job definition and the resulting data."""

    # Generic fields
    status: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # Pipeline fields
    setup: Setup | None = None
    data: List[Document] = field(default_factory=list)

    @property
    def _id_prefix(self) -> str:
        return "spk_"

    def runtime(self) -> timedelta | None:
        if self.started_at is not None:
            if self.finished_at is None:
                return datetime.now() - self.started_at
            return self.finished_at - self.started_at
        return None


SpoCKList = List[SpoCK]


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for the :class:`Document` class."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Document):
            return asdict(obj)
        if isinstance(obj, np.float32):
            return str(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class FileHandler:
    """Reads from and write data to a JSON file under a given file path."""

    _suffix = ".json"

    def __init__(self, file_path: Path | str) -> None:
        self._file_path = Path(file_path) if isinstance(file_path, str) else file_path
        if not self._file_path.exists():
            os.makedirs(self._file_path)

    def read(self, filename: str) -> List[Document]:
        """Reads the data from a JSON file and casts it into a list of :class:`Document` objects."""
        filename = (self._file_path / filename).with_suffix(self._suffix)

        logger.info("reading data from file {filename}")
        with open(filename.with_suffix(self._suffix), "r", encoding="utf-8") as dfile:
            dict_ = json.load(dfile)

        return SpoCK.from_dict(dict_=dict_)

    def write(self, file_name: str, spock: SpoCK, add_dt: bool = True) -> None:
        """Writes the data to a JSON file.

        If `add_dt=True`, the filename is `{file_name}_%Y%m%d%H%M%S.json`.
        """
        if add_dt:
            file_name = f"{file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        file_name = (self._file_path / file_name).with_suffix(self._suffix)

        logger.info(f"writing data to file {file_name}")
        with open(file_name, "w", encoding="utf-8") as dfile:
            json.dump(spock.to_dict(), dfile, cls=JSONEncoder)
