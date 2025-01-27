from abc import ABC, abstractmethod
import aiohttp
from argparse import Namespace
import asyncio
import logging
import os
import re
from typing import List

from openai import AsyncOpenAI

from vianu.spock.src.base import NamedEntity, QueueItem  # noqa: F401
from vianu.spock.settings import (
    N_CHAR_DOC_ID,
    LLAMA_MODEL,
    OPENAI_MODEL,
    MODEL_TEST_QUESTION,
)

logger = logging.getLogger(__name__)

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
    _named_entity_pattern = re.compile(r'\("([^"]+)",\s?"(MP|ADR)"\)')

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _get_loc_of_subtext(text: str, subtext: str) -> List[int] | None:
        """Get the location of a subtext in a text."""
        pos = text.find(subtext)
        if pos == -1:
            return None
        return [pos, pos + len(subtext)]

    def _add_loc_for_named_entities(
        self, text: str, named_entities: List[NamedEntity]
    ) -> None:
        txt_low = text.lower()
        for ne in named_entities:
            ne_txt_low = ne.text.lower()
            loc = self._get_loc_of_subtext(text=txt_low, subtext=ne_txt_low)

            if loc is not None:
                ne.location = loc
                ne.text = text[loc[0] : loc[1]]
            else:
                self.logger.warning(
                    f'could not find location for named entity "{ne.text}" of class "{ne.class_}"'
                )

    @staticmethod
    def _get_messages(text: str) -> List[dict]:
        text = f'Process the following input text: "{text}"'
        return [
            {
                "role": "system",
                "content": NAMED_ENTITY_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

    @staticmethod
    def _get_test_messages(text: str) -> List[dict]:
        return [
            {"role": "system", "content": "Answer very briefly."},
            {
                "role": "user",
                "content": text,
            },
        ]

    @abstractmethod
    async def _get_model_answer(self, text: str) -> str:
        raise NotImplementedError(
            "OpenAINER._get_ner_model_answer is not implemented yet"
        )

    @abstractmethod
    async def test_model_endpoint(self) -> str:
        pass

    async def apply(self, queue_in: asyncio.Queue, queue_out: asyncio.Queue) -> None:
        """Apply NER to a text received from input queue and put the results in an output queue."""

        while True:
            # Get text from input queue
            item = await queue_in.get()  # type: QueueItem

            # Check stopping condition
            if item is None:
                queue_in.task_done()
                break

            # Get the model response with named entities
            id_ = item.id_
            doc = item.doc
            self.logger.debug(
                f"starting ner for item.id_={id_} (doc.id_={doc.id_[:N_CHAR_DOC_ID]})"
            )
            try:
                text = doc.text
                content = await self._get_model_answer(text=text)
            except Exception as e:
                self.logger.error(
                    f"error during ner for item.id_={item.id_} (doc.id_={doc.id_[:N_CHAR_DOC_ID]}): {e}"
                )
                queue_in.task_done()
                continue

            # Parse the model answer and remove duplicates
            ne_list = re.findall(self._named_entity_pattern, content)
            ne_list = list(set(ne_list))

            # Create list of NamedEntity objects
            named_entities = []
            for ne in ne_list:
                try:
                    txt, cls_ = ne
                    named_entities.append(
                        NamedEntity(id_=f"{text} {txt} {cls_}", text=txt, class_=cls_)
                    )
                except Exception as e:
                    self.logger.error(
                        f"error during creation of `NamedEntity` using {ne}: {e}"
                    )

            # Add locations to named entities and remove those without location
            self._add_loc_for_named_entities(text=text, named_entities=named_entities)
            named_entities = [ne for ne in named_entities if ne.location is not None]

            # Assign named entities to the document
            ne_mp = [ne for ne in named_entities if ne.class_ == "MP"]
            ne_adr = [ne for ne in named_entities if ne.class_ == "ADR"]
            self.logger.debug(
                f"found #mp={len(ne_mp)} and #adr={len(ne_adr)} for item.id_={id_} (doc.id_={doc.id_[:N_CHAR_DOC_ID]})"
            )
            doc.medicinal_products = ne_mp
            doc.adverse_reactions = ne_adr

            # Put the document in the output queue
            await queue_out.put(item)
            queue_in.task_done()
            self.logger.info(
                f"finished NER task for item.id_={id_} (doc.id_={doc.id_[:N_CHAR_DOC_ID]})"
            )


class OpenAINER(NER):
    def __init__(self, model: str, api_key: str):
        super().__init__()
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def _chat_completion(self, messages: List[dict]) -> str:
        chat_completion = await self._client.chat.completions.create(
            messages=messages,
            model=self._model,
        )
        return chat_completion.choices[0].message.content

    async def _get_model_answer(self, text: str) -> str:
        messages = self._get_messages(text=text)
        return await self._chat_completion(messages=messages)

    async def test_model_endpoint(self) -> str:
        messages = self._get_test_messages(text=MODEL_TEST_QUESTION)
        return await self._chat_completion(messages=messages)


class OllamaNER(NER):
    def __init__(self, model: str, base_url: str):
        super().__init__()
        self._model = model
        self._base_url = base_url

    def _get_http_data(self, text: str, stream: bool = False) -> dict:
        messages = self._get_messages(text=text)
        data = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        return data

    async def _get_model_answer(self, text: str) -> str:
        data = self._get_http_data(text=text)
        async with aiohttp.ClientSession() as session:
            url = f"{self._base_url}/api/chat/"
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                resp_json = await response.json()
                content = resp_json["message"]["content"]
        return content

    async def test_model_endpoint(self) -> str:
        messages = self._get_test_messages(text=MODEL_TEST_QUESTION)
        data = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        async with aiohttp.ClientSession() as session:
            url = f"{self._base_url}/api/chat/"
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                resp_json = await response.json()
                content = resp_json["message"]["content"]
        return content


class NERFactory:
    """Factory for NER models."""

    @staticmethod
    def create(model: str, config: dict) -> NER:
        """Create a NER model from the `model` keyword."""

        # create a :class:`OllamaNER` instance
        if model == "openai":
            api_key = config[model].get("api_key")
            if api_key is None:
                api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "The api_key for the OpenAI client is missing (set it by the OPENAI_API_KEY environment variable or in the settings)"
                )
            ner = OpenAINER(model=OPENAI_MODEL, api_key=api_key)

        elif model == "llama":
            base_url = config[model].get("base_url")
            if base_url is None:
                base_url = os.environ.get("OLLAMA_BASE_URL")
            if base_url is None:
                raise ValueError("The base_url for the ollama endpoint is missing")
            ner = OllamaNER(model=LLAMA_MODEL, base_url=base_url)
        else:
            raise ValueError(f"unknown ner model '{model}'")

        return ner


def create_tasks(
    args_: Namespace,
    queue_in: asyncio.Queue,
    queue_out: asyncio.Queue,
    model_config: dict,
) -> List[asyncio.Task]:
    """Create asyncio NER tasks."""
    n_ner_tasks = args_.n_ner_tasks
    model = args_.model
    ner = NERFactory.create(model=model, config=model_config)

    logger.info(f"setting up {n_ner_tasks} NER task(s)")
    tasks = [
        asyncio.create_task(ner.apply(queue_in=queue_in, queue_out=queue_out))
        for _ in range(n_ner_tasks)
    ]
    return tasks
