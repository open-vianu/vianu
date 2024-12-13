from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import ast
from copy import deepcopy
import logging
import re
import requests
from typing import List

from ..settings import DEFAULT_OLLAMA_ENDPOINT, DEFAULT_LLAMA_MODEL
from .data_model import Document, FileHandler, NamedEntity

logger = logging.getLogger(__name__)

MODULE_NAME = 'ner'

NAMED_ENTITY_PROMPT = """
You are an expert in Natural Language Processing. Your task is to identify named entities (NER) in a given text.
You will focus on the following entities: adverse drug reaction (entity type: ADR), medicinal product (entity type: MP).
Once you identified all named entities of the above types, you return them as a Python list of tuples of the form (text, type). 
It is important to only provide the Python list as your output, without any additional explanations or text.
In addition, make sure that the named entity texts are exact copies of the original text segment

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
"""


class NER(ABC):

    @staticmethod
    def get_loc_of_subtext(text: str, subtext: str) -> List[int] | None:
        """Get the location of a subtext in a text."""
        pos = text.find(subtext)
        if pos == -1:
            return None
        return pos, pos + len(subtext)


    @abstractmethod
    def apply(text: str) -> List[NamedEntity]:
        pass




class OllamaNER(NER):


    def __init__(self, endpoint: str, model: str):
        self._endpoint = endpoint
        self._model = model


    def _get_ner_data(self, text: str, stream: bool = False) -> dict:
        user_text = f'Process the following input text: "{text}"'
        data = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": NAMED_ENTITY_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            "stream": stream,
        }
        return data
    

    def _get_ner_model_answer(self, text: str, stream: bool = False) -> str:
        data = self._get_ner_data(text=text, stream=stream)
        response = requests.post(self._endpoint, json=data, stream=stream)
        response.raise_for_status()
        content = response.json()['message']['content']
        return content


    def _add_loc_for_named_entities(self, text: str, named_entities: List[NamedEntity]) -> None:
        txt_low = text.lower()
        for ne in named_entities:
            ne_txt_low = ne.text.lower()
            loc = self.get_loc_of_subtext(text=txt_low, subtext=ne_txt_low)

            if loc is not None:
                ne.location = loc
                ne.text = text[loc[0]:loc[1]]
            else:
                logger.warning(f'could not find location for named entity "{ne.text}" of class "{ne.class_}"')


    def apply(self, text: str) -> List[NamedEntity]:
        # Request the text answer from the model endpoint and transforms it into a list of NamedEntity objects
        try: 
            content = self._get_ner_model_answer(text=text)
        except Exception as e:
            logger.error(f'error during inference of model endpoint: {e}')
            logger.debug(f'text: {text}')
            return []
        
        # Parse the model answer and remove duplicates
        ne_list = re.findall(r'\("([^"]+)",\s?"(MP|ADR)"\)', content)
        ne_list = list(set(ne_list))

        # Create list of NamedEntity objects
        named_entities = []
        for ne in ne_list:
            try:
                txt, cls_ = ne
                named_entities.append(NamedEntity(id_=f'{text} {txt} {cls_}', text=txt, class_=cls_))
            except Exception as e:
                logger.error(f'error during creation of NamedEntity object: {e}')
                logger.debug(f'ne={ne}')

        # Add locations to named entities
        self._add_loc_for_named_entities(text=text, named_entities=named_entities)
        return [ne for ne in named_entities if ne.location is not None]


def cli_args() -> None:    
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--model', dest='model', choices=['ollama'], default='ollama')
    return parser


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


    if save_data:
        FileHandler(args_.data_dump).write(data)
