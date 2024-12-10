from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import ast
from copy import deepcopy
import logging
import requests
from typing import List

from ..settings import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_LLAMA_MODEL
from .data_model import Document, FileHandler, NamedEntity

logger = logging.getLogger(__name__)

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

    @staticmethod
    def get_loc_of_subtext(text: str, sutext: str) -> List[int] | None:
        """Get the location of a subtext in a text."""
        pos = text.find(sutext)
        if pos == -1:
            return None
        return pos, pos + len(sutext)


    @abstractmethod
    def apply(text: str) -> List[NamedEntity]:
        pass




class OllamaNER(NER):

    def __init__(self, endpoint: str, model: str):
        self._data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": NAMED_ENTITY_PROMPT,
                },
            ],
            "stream": False,
        }
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
        

    def apply(self, text: str) -> List[NamedEntity]:
        # Request the text answer from the model endpoint
        try: 
            content = self._get_model_answer(text=text)
        except Exception as e:
            logger.error(f'error during inference of model endpoint: {e}')
            logger.debug(f'text: {text}')
            return []
        
        # Parse the model answer
        try:
            ne_list = ast.literal_eval(content)
            if not isinstance(ne_list, list):
                raise TypeError('parsed model answer is not a list')
        except Exception as e:
            logger.error(f'error while parsing model answer: {e}')
            logger.debug(f'content: {content}')
        
        named_entities = []
        for ne in ne_list:
            try:
                txt, cls_ = ne
                named_entities.append(NamedEntity(id_=f'{text} {txt} {cls_}', text=txt, class_=cls_))
            except Exception as e:
                logger.error(f'error during creation of NamedEntity object: {e}')
                logger.debug(f'ne={ne}')
        return named_entities



def cli_args() -> None:    
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--model', dest='model', choices=['ollama'], default='ollama')
    return parser


def _add_loc_for_medicinal_products(doc: Document) -> None:
    txt_low = doc.text.lower()
    for ne in doc.medicinal_products:
        ne_txt_low = ne.text.lower()
        loc = NER.get_loc_of_subtext(text=txt_low, sutext=ne_txt_low)
        if loc is not None:
            ne.location = loc
            ne.text = doc.text[loc[0]:loc[1]]
        else:
            logger.warning(f'could not find location for medicinal product "{ne.text}"')


def _add_loc_for_adverse_reactions(doc: Document) -> None:
    txt_low = doc.text.lower()
    for ne in doc.adverse_reactions:
        ne_txt_low = ne.text.lower()
        loc = NER.get_loc_of_subtext(text=txt_low, sutext=ne_txt_low)
        if loc is not None:
            ne.location = loc
            ne.text = doc.text[loc[0]:loc[1]]
        else:
            logger.warning(f'could not find location for adverse reaction "{ne.text}"')

def _postprocess_named_entities(data: List[Document]) -> None:
    for doc in data:
        _add_loc_for_medicinal_products(doc)
        _add_loc_for_adverse_reactions(doc)
    

def apply(args_: Namespace, data: List[Document] | None = None, save_data: bool = True) -> None:
    if data is None:
        data = FileHandler(args_.data_load).read()
    
    data_no_err = [d for d in data if not d.has_error()]
    logger.info(f'NER for {len(data_no_err)} documents')
    
    if args_.model == 'ollama':
        ner = OllamaNER(endpoint=DEFAULT_OLLAMA_ENDPOINT, model=DEFAULT_LLAMA_MODEL)
    else:
        logger.error(f'unknown ner model "{args_.model}"')
        return None
    

    for id, doc in enumerate(data_no_err):
        logger.debug(f'NER for document #{id} with id={doc.id_}')
        named_entities = ner.apply(text=doc.text)
        
        # Add medicinal products
        doc.medicinal_products.extend([ne for ne in named_entities if ne.class_ == 'MP'])
        logger.debug(f'found {len(doc.medicinal_products)} medicinal products')
        
        # Add adverse reactions
        doc.adverse_reactions.extend([ne for ne in named_entities if ne.class_ == 'ADR'])
        logger.debug(f'found {len(doc.adverse_reactions)} adverse reactions')

    # TODO this must change!!!! include it into the NER class
    logger.info('cleaning up named entities')
    _postprocess_named_entities(data)

    if save_data:
        FileHandler(args_.data_dump).write(data)
