import json
from typing import Dict

from llama_cpp import Llama

import settings
from helpers import llm_apis


class BaseGenerator:
    def __init__(self, prompt: str, config: dict, exclude_keys: list = ["root_content_id"]):
        self.prompt = prompt
        self.config = config
        self.exclude_keys = exclude_keys

    def _clean_dict(self, d: dict) -> dict:
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, str) and not k.startswith("_") and k not in self.exclude_keys:
                new_dict[k] = v
        return new_dict

    def _clean_chunks(self, chunks: Dict[str, dict]) -> Dict[str, dict]:
        chunks_cln = []
        for chunk in chunks.values():
            chunk_cln = self._clean_dict(chunk)
            chunks_cln.append(chunk_cln)

        return chunks_cln

    def _make_prompt(self, question: str, chunks: Dict[str, dict]) -> str:
        chunks_cln = self._clean_chunks(chunks)
        context_str = "\n".join([json.dumps(chunk, indent=4) for chunk in chunks_cln])
        prompt = self.prompt.format(
            question=question,
            retrieved_chunks=context_str,
        )

        return prompt

    def _parse_response(self, response: dict) -> dict:
        try:
            content = response["content"].replace("```json\n", "").replace("```", "")
            response_dict = json.loads(content)
            answer = response_dict["answer"]
            chunk_ids = response_dict["chunk_ids"]
            chunk_ids = [str(i) for i in chunk_ids]
        except json.JSONDecodeError:
            answer = response["content"]
            chunk_ids = None

        return answer, chunk_ids


class LocalGenerator(BaseGenerator):
    def __init__(self, prompt: str, config: dict, exclude_keys: list = ["root_content_id"]):
        super().__init__(prompt, config, exclude_keys)
        self.local_llm = self.load_local_model(settings.LOCAL_LLM)

    def load_local_model(self, model_name):
        max_context_length = self.config.get("max_context_length", 512)
        return Llama(model_path=f"models/{model_name}.gguf", n_ctx=max_context_length)

    def generate(self, question: str, chunks: Dict[str, dict]) -> str:
        prompt = self._make_prompt(question, chunks)
        response = self.local_llm(
            prompt,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0.8),
            top_p=self.config.get("top_p", 0.95),
            min_p=self.config.get("min_p", 0.05),
            top_k=self.config.get("top_k", 40),
            seed=self.config.get("seed", None),
            stop=self.config.get("stop", []),
        )["choices"][0]
        answer = response["text"]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks.values()]

        return answer, chunk_ids, prompt


class RemoteGenerator(BaseGenerator):
    def __init__(self, prompt: str, config: dict, exclude_keys: list = ["root_content_id"]):
        super().__init__(prompt, config, exclude_keys)
        self.llm_api = llm_apis.MistralAPI()

    def generate(self, question: str, chunks: Dict[str, dict]) -> str:
        prompt = self._make_prompt(question, chunks)
        response = self.llm_api.call_api(prompt, self.config)
        answer, chunk_ids = self._parse_response(response)

        return answer, chunk_ids, prompt
