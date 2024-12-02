from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import logging
from typing import List

from .data_model import Document, FileHandler, NamedEntity

MODULE_NAME = 'ner'


class NERecognizer(ABC):

    @abstractmethod
    def get_named_entities(text: str) -> List[NamedEntity]:
        pass




def cli_args() -> None:    
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    return parser



def apply(args_: Namespace, data: List[Document] | None = None) -> None:

    if data is None:
        data = FileHandler(args_.data_load).read()
    
    data_no_err = [d for d in data if not d.has_error()]
    logging.info(f'extracting named entities for {len(data_no_err)} documents')
    
    if args_.model == 'ollama':
        pass
        
    else:
        logging.error(f'unknown ner model "{args_.model}"')
        return None
    

    for doc in data_no_err:
        logging.debug(f'performing ner for {len(doc.text_entities)} text entities in document {doc.id_}')
        for te in doc.text_entities:
            ner = client.ner(text=te.text)
            logging.debug(ner)
        logging.debug(f'found {sum([len(te.medicinalProducts) for te in doc.text_entities])} medicinal products in document {doc.id_}')
        logging.debug(f'found {sum([len(te.signals) for te in doc.text_entities])} signals in document {doc.id_}')
    
    FileHandler(args_.data_dump).write(data)