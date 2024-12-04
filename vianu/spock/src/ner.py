from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import ast
from copy import deepcopy
import logging
import requests
from typing import List

from ..settings import DEFAULT_OLLAMA_ENDPOINT
from .data_model import Document, TextEntity, FileHandler, NamedEntity
from . import utils as utl

MODULE_NAME = 'ner'
NAMED_ENTITY_PROMPT = """
You are an expert assistant trained to extract specific information from text. Given the following text, return a 
Python list of tuples of all named entities together with their type mentioned in the text. You will focus on the 
following entities: adverse drug reaction (entity type: ADR), medicinal product (entity type: MP).
Provide only the Python list as your output, without any additional explanations or text.

Example 1:
Input:
"The most commonly reported side effects of dafalgan include headache, nausea, and fatigue."

Output:
[("dafalgan", "MP"), ("headache", "ADR"), ("nausea", "ADR"), ("fatigue", "ADR")]

Example 2:
Input:
"Patients taking acetaminophen or naproxen have reported experiencing skin rash, dry mouth, and difficulty breathing after taking this medication. In rare cases, seizures have also been observed."

Output:
[("acetaminophen", "MP"), ("naproxen", "MP"), ("skin rash", "ADR"), ("dry mouth", "ADR"), ("difficulty breathing", "ADR"), ("seizures", "ADR")]

Example 3:
Input:
"There are reported side effects as dizziness, stomach upset, and in some instances, temporary memory loss. These are mainly observed after taking Amitiza (lubiprostone) or Trulance (plecanatide)."

Output:
[("dizziness", "ADR"), ("stomach upset", "ADR"), ("temporary memory loss", "ADR"), ("Amitiza (lubiprostone)", "MP"), ("Trulance (plecanatide)", "MP")]

Now, analyze the following text and return a Python list of all adverse events and side effects:
"""


class NER(ABC):


    @abstractmethod
    def get_named_entities(text: str) -> List[NamedEntity]:
        pass


class OllamaNER(NER):
    _data = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": NAMED_ENTITY_PROMPT,
            },
        ],
        "stream": False,
    }

    def __init__(self, endpoint: str):
        super(OllamaNER, self).__init__()
        self._endpoint = endpoint


    def _get_model_answer(self, text: str) -> str:
        data = deepcopy(self._data)
        data['messages'].append({
            "role": "user",
            "content": text,
        })
        response = requests.post(self._endpoint, json=data, stream=False)
        response.raise_for_status()
        content = response.json()['message']['content']

        return content
        

    def get_named_entities(self, text: str) -> List[NamedEntity]:
        # Request the text answer from the model endpoint
        try: 
            content = self._get_model_answer(text=text)
        except Exception as e:
            logging.error(f'error during inference of model endpoint: {e}')
            logging.debug(f'text: {text}')
            return []
        
        # Parse the model answer
        ne_list = ast.literal_eval(content)
        if not isinstance(ne_list, list):
            logging.error(f'error while parsing model answer: content={content}')
            return []
        
        named_entities = []
        for ne in ne_list:
            try:
                txt, cls_ = ne
                named_entities.append(NamedEntity(id_=f'{text} {txt} {cls_}', text=txt, class_=cls_))
            except Exception as e:
                logging.error(f'error during creation of NamedEntity object: {e}')
                logging.debug(f'ne={ne}')
        return named_entities



def cli_args() -> None:    
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--model', dest='model', choices=['ollama'], default='ollama')
    return parser


def _add_loc_for_medicinal_products(text_entity: TextEntity) -> None:
    te_txt_low = text_entity.text.lower()
    for ne in text_entity.medicinal_products:
        ne_txt_low = ne.text.lower()
        loc = utl.get_loc_of_subtext(text=te_txt_low, sutext=ne_txt_low)
        if loc is not None:
            ne.location = loc
            ne.text = text_entity.text[loc[0]:loc[1]]


def _add_loc_for_adverse_reactions(text_entity: TextEntity) -> None:
    te_txt_low = text_entity.text.lower()
    for ne in text_entity.adverse_reactions:
        ne_txt_low = ne.text.lower()
        loc = utl.get_loc_of_subtext(text=te_txt_low, sutext=ne_txt_low)
        if loc is not None:
            ne.location = loc
            ne.text = text_entity.text[loc[0]:loc[1]]


def _postprocess_named_entities(data: List[Document]) -> None:
    for doc in data:
        for te in doc.text_entities:
            _add_loc_for_medicinal_products(te)
            _add_loc_for_adverse_reactions(te)
    

def apply(args_: Namespace, data: List[Document] | None = None, save_data: bool = True) -> None:
    if data is None:
        data = FileHandler(args_.data_load).read()
    
    data_no_err = [d for d in data if not d.has_error()]
    logging.info(f'extracting named entities for {len(data_no_err)} documents')
    
    if args_.model == 'ollama':
        client = OllamaNER(endpoint=DEFAULT_OLLAMA_ENDPOINT)
    else:
        logging.error(f'unknown ner model "{args_.model}"')
        return None
    

    for id, doc in enumerate(data_no_err):
        logging.debug(f'NER for {len(doc.text_entities)} text entities in document {id} with id {doc.id_}')
        for jt, te in enumerate(doc.text_entities):
            logging.debug(f'text entity {jt} with id_={te.id_}')
            named_entities = client.get_named_entities(text=te.text)
            te.medicinal_products.extend([ne for ne in named_entities if ne.class_ == 'MP'])
            te.adverse_reactions.extend([ne for ne in named_entities if ne.class_ == 'ADR'])
            logging.debug(f'found {len(te.medicinal_products)} medicinal products')
            logging.debug(f'found {len(te.adverse_reactions)} adverse reactions')

    logging.info('cleaning up named entities')
    _postprocess_named_entities(data)

    if save_data:
        FileHandler(args_.data_dump).write(data)
