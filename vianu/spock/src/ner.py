from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
import ast
import json
import logging
import requests
from typing import List, Tuple

from .data_model import Document, FileHandler, NamedEntity
from ..settings import DEFAULT_OLLAMA_ENDPOINT

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
        data = self._data.copy()
        data['messages'].append({
            "role": "user",
            "content": text,
        })
        response = requests.post(self._endpoint, json=data, stream=False)
        response.raise_for_status()
        content = response.json()['message']['content']

        # content = ''
        # for line in response.iter_lines():
            # if line:
                # data = json.loads(line.decode('utf-8'))
                # content += data['message']['content']
                # if data['done']:
                    # break
        return content
        

    def get_named_entities(self, text: str) -> List[NamedEntity]:
        try:
            content = self._get_model_answer(text=text)
            named_entities = ast.literal_eval(content)
            return [NamedEntity(id_=f'{text} {txt} {cls_}', text=txt, class_=cls_) for txt, cls_ in named_entities]
        except Exception as e:
            logging.error(f'error while extracting named entities: {e}')
            return []



def cli_args() -> None:    
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--model', dest='model', choices=['ollama'], default='ollama')
    return parser



def apply(args_: Namespace, data: List[Document] | None = None) -> None:

    if data is None:
        data = FileHandler(args_.data_load).read()
    
    data_no_err = [d for d in data if not d.has_error()]
    logging.info(f'extracting named entities for {len(data_no_err)} documents')
    
    if args_.model == 'ollama':
        client = OllamaNER(endpoint=DEFAULT_OLLAMA_ENDPOINT)
    else:
        logging.error(f'unknown ner model "{args_.model}"')
        return None
    

    for doc in data_no_err:
        logging.debug(f'performing ner for {len(doc.text_entities)} text entities in document {doc.id_}')
        for i, te in enumerate(doc.text_entities):
            logging.debug(f'text entity {i} with id_={te.id_}')
            named_entities = client.get_named_entities(text=te.text)
            te.medicinal_products.extend([ne for ne in named_entities if ne.class_ == 'MP'])
            te.adverse_reactions.extend([ne for ne in named_entities if ne.class_ == 'ADR'])
            logging.debug(f'found {len(te.medicinal_products)} medicinal products')
            logging.debug(f'found {len(te.adverse_reactions)} adverse reactions')
    
    FileHandler(args_.data_dump).write(data)