from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from hashlib import sha256
from typing import List


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
class NamedEntityUnit(DataUnit):
    """Parent class for all named entities."""
    text: str = field(default_factory=str)
    location: List[int] | None = None
    probability: float | None = None

    @property
    def _id_prefix(self):
        return 'ent_'


@dataclass(eq=False)
class MedProdSigPair(DataUnit):
    """Class describing a medicinal product - signal pair named entity."""

    medProdID: str = field(default_factory=str)
    sigID: str = field(default_factory=str)
    causality: float | None = None
    seriousness: float | None = None
    severity: float | None = None

    @property
    def _id_prefix(self):
        return 'msp_'


@dataclass(eq=False)
class TextEntity(DataUnit):
    """Class containing any text-entity related information."""

    # Text information
    text: str = field(default_factory=str)
    location: List[int] = field(default_factory=list)

    # NER information (optional)
    medicinalProducts: List[NamedEntityUnit] = field(default_factory=list)
    signals: List[NamedEntityUnit] = field(default_factory=list)
    medProdSigPairs: List[MedProdSigPair] = field(default_factory=list)


    # private document fields
    __raw_text: str = ""

    @property
    def _id_prefix(self):
        return 'txt_'

    @staticmethod
    def _get_filtered_entities(entities) -> list[str]:
        ref = [x.text.strip().lower().strip("-").strip() for x in entities]
        return list(set([x for x in ref if len(x) > 1]))

    def get_products(self) -> list[str]:
        return self._get_filtered_entities(self.medicinalProducts)

    def get_signals(self) -> list[str]:
        return self._get_filtered_entities(self.signals)

    def set_med_prod_sig_pair_causality(
            self,
            causality: float,
            index_: int | None = None,
            id_: str | None = None
    ) -> None:
        """Set the causality value for a given medicinal product - signal pair.

        Args:
            causality (float): Causality value in [0, 1]
            index (int): List index of the medicinal product - signal pair.
            id_ (str): ID of the medicinal product - signal pair.
        """
        if not 0 <= causality <= 1:
            raise ValueError(f'Causality value={causality} is not contained in [0, 1]')
        if index_ is not None and id_ is not None:
            raise ValueError(f'The definition of `index` OR `id_` must be exclusive.')
        if index_ is None and id_ is None:
            raise ValueError(f'Either `index` or `id_` has to be defined.')
        if id_ is not None:
            id2index = {p.id_: i for i, p in enumerate(self.medProdSigPairs)}
            index = id2index[id_]
        elif index_ is not None:
            index = index_
        self.medProdSigPairs[index].causality = causality

    def add_raw_text(self, text: str) -> None:
        self.__raw_text = text

    def get_raw_text(self) -> str:
        return self.__raw_text

# TODO continue from here

@dataclass(eq=False)
class Document(DataUnit):
    """Class containing any document related information."""

    url: str | None = None
    sourceUrl: str | None = None
    language: str | None = None

    # optional document fields
    text: str | None = None
    title: str | None = None
    abstract: str | None = None

    # protected document fields (considered by `DataUnit.asdict()`)
    _textEntities: List[TextEntity] = field(default_factory=list)
    _errors: List[str] = field(default_factory=list)

    # private document fields
    __raw_text: str = ""

    @property
    def _id_prefix(self):
        return 'doc_'

    @property
    def textEntities(self):
        return self._textEntities

    def add_text_entity(self, text_entity: TextEntity) -> None:
        """Appends a text entity and updates the ID - Index map in self.textEntityIDIndexMap."""
        if (id_ := text_entity.id_) in self.__textEntityIDIndexMap:
            index = self.__textEntityIDIndexMap[id_]
            logging.debug(f'A textEntity object with id={id_} already exists at loc={index} and is ignored')
        else:
            self.textEntities.append(text_entity)
            self.__textEntityIDIndexMap[text_entity.id_] = len(self.textEntities) - 1

    def del_text_entity(self, index_: int | None = None, id_: str | None = None) -> None:
        """Removes a given text entity by its index or its id."""
        if index_ is not None and id_ is not None:
            raise ValueError(f'The definition of `index` OR `id_` must be exclusive.')
        if index_ is None and id_ is None:
            raise ValueError(f'Either `index` or `id_` has to be defined.')
        if id_ is not None:
            index = self.__textEntityIDIndexMap[id_]
        elif index_ is not None:
            index = index_
        self._textEntities.pop(index)
        self.__textEntityIDIndexMap = {ent.id_: i for i, ent in enumerate(self._textEntities)}

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

    def add_input(self, input: str) -> None:
        self.__input = input

    def get_input(self) -> str | None:
        return self.__input

    def is_source_binary(self) -> bool:
        url = ("." + self.type.value) if self.type else (self.url or "")
        return any([url.endswith(x) for x in [".xls", ".xlsx", ".pdf", ".ppt"]])

    def is_source_html(self) -> bool:
        return self.type is not None and self.type == DocumentTypes.HTML or (self.url or "").endswith(".html")

    def _apply_feedback(self, feedback: Feedback) -> None:
        match feedback.category:
            case FeedbackCategories.NER:
                def _create_instance(class_name, id_, text, location, probability):
                    cls = globals()[class_name]
                    instance = cls(id_=id_, text=text, location=location, probability=probability)
                    return instance
                
                old_value = json.loads(str(feedback.old_value))
                new_value = json.loads(feedback.new_value)

                for entity in self.textEntities:
                    if entity.id_ != feedback.text_entity_id:
                        continue
                    filtered = [pair for pair in getattr(entity, old_value["key"]) if pair.id_ != old_value["entity_id"]]
                    setattr(entity, old_value["key"], filtered)

                    if new_value["key"] != "removed":
                        new_pair = _create_instance(new_value["class_name"], id_=new_value["entity_id"], text=new_value["text"], location=new_value["location"], probability=float(new_value["probability"]))
                        getattr(entity, new_value["key"]).append(new_pair)
                        
                    if old_value["key"] == "medicinalProducts":
                        entity.medProdSigPairs = [pair for pair in entity.medProdSigPairs if pair.medProdID != old_value["entity_id"]]
                    if old_value["key"] == "signals":
                        entity.medProdSigPairs = [pair for pair in entity.medProdSigPairs if pair.sigID != old_value["entity_id"]]

            case FeedbackCategories.CAUSALITY:
                for entity in self.textEntities:
                    if entity.id_ != feedback.text_entity_id:
                        continue
                    for pair in entity.medProdSigPairs:
                        pair.causalityFeedback = float(feedback.new_value)
            case FeedbackCategories.SEVERITY:
                for entity in self.textEntities:
                    if entity.id_ != feedback.text_entity_id:
                        continue
                    for pair in entity.medProdSigPairs:
                        pair.severityFeedback = float(feedback.new_value)
            case FeedbackCategories.RELEVANCE:
                for entity in self.textEntities:
                    if entity.id_ != feedback.text_entity_id:
                        continue
                    for pair in entity.medProdSigPairs:
                        pair.relevanceFeedback = float(feedback.new_value)
            case FeedbackCategories.REPORT:
                new_value = json.loads(feedback.new_value)
                if new_value["enabled"]:
                    if new_value["query_id"] not in self.marked_for_report:
                        self.marked_for_report.append(new_value["query_id"])
                elif new_value["query_id"] in self.marked_for_report:
                    self.marked_for_report.remove(new_value["query_id"])
            case FeedbackCategories.COMMENT:
                self.comment = feedback.new_value
            case FeedbackCategories.CONTEXT:
                new_value = json.loads(feedback.new_value)
                for entity in self.textEntities:
                    if entity.id_ != feedback.text_entity_id:
                        continue
                    if new_value["enabled"]:
                            entity.reportContext = new_value["context"]
                    else:
                        entity.reportContext = []

    def apply_feedbacks(self, feedbacks: list[Feedback] | None) -> None:
        if not feedbacks:
            return
        logging.warning(f"Applying {len(feedbacks)} on doc {self.id_}")
        for feedback in feedbacks:
            self._apply_feedback(feedback)

    def set_last_update(self, previous: Document | None):
        """Set lastUpdate fields based on previous version of same document"""

        if previous is not None and previous.lastUpdate is not None and self.lastUpdate is None:
            self.lastUpdate = previous.lastUpdate

        if self.lastUpdate is None:
            self.lastUpdate = self.extractionDate

        if previous is None:
            logging.info(f"No previous version of {self.id_}")
            return

        changed = False
        logging.info(f"Comparing with previous version {self.id_}")
        for index, textEntity in enumerate(self.textEntities):
            if textEntity.lastUpdate is None:
                logging.info(f"{self.id_}: textEntity lastUpdate was None, setting to {self.lastUpdate}")
                self._textEntities[index].lastUpdate = self.lastUpdate
            # Find in previous
            found = False
            for prev_entity in previous.textEntities:
                if prev_entity.id_ != textEntity.id_:
                    continue
                # Found an entity that existed before
                found = True
                # TODO this will always return False (dataclass eq=False)
                # we might want to consider that some changes should be notified to users !
                if textEntity != prev_entity:
                    changed = True
                    logging.warning(f"Doc {self.id_} entity {textEntity.id_} has a change")
                    self._textEntities[index].lastUpdate = datetime.now()
            if not found:
                # New entity
                logging.info(f"{self.id_}: found new entity")
                self._textEntities[index].lastUpdate = datetime.now()
                changed = True
        if changed:
            logging.info(f"{self.id_}: doc changed, setting new lastUpdate")
            self.lastUpdate = datetime.now()


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
        if isinstance(o, Enum):
            return o.value
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


class DocumentList(List[Document]):
    def _dump(self, dest: IO[str]):
        logging.debug(f"Saving {len(self)} entries")
        json.dump(self, dest, cls=DocumentJSONEncoder)

    def dumps(self):
        stream = io.StringIO()
        self._dump(stream)
        return stream.getvalue()

    def dump(self, path: Path):
        with open(path, 'w', encoding="utf-8") as pfile:
            self._dump(pfile)

    def load_json(self, entries: List[Dict]):
        for x in entries:
            self.append(dacite.from_dict(data_class=Document, data=x, config=dacite.Config(cast=[Enum], type_hooks={datetime: datetime.fromisoformat})))

    def load(self, path: Path):
        with open(path, 'r', encoding="utf-8") as pfile:
            entries = json.load(pfile)
            assert isinstance(entries, list)
            self.load_json(entries)

    def set_docs_language(self, language:str):
        logging.warning(f"Setting language to {language}")
        for doc in self:
            doc.language = language

    def get_doc_from_id(self, id_: str) -> Document | None:
        for doc in self:
            if doc.id_ == id_:
                return doc
        return None

    def get_doc_from_url(self, url: str) -> Document | None:
        for doc in self:
            if str(doc.url) == url:
                return doc
        return None