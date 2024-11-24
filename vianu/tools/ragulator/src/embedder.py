import json
import logging
import os
from typing import Dict

import click
import numpy as np
from sentence_transformers import SentenceTransformer

import settings
import llm_apis

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """
    Base class with common methods for embedding chunks and questions.
    """

    def embed_chunk(self, chunk: Dict, output_type: str) -> np.ndarray:
        """
        Embed a chunk of content.
        """
        chunk_content = ""
        for key in settings.EMBEDDING_KEYS:
            if key in chunk:
                chunk_content += f"{chunk[key]}\n"
        chunk_embedding = self._embed(chunk_content)
        if output_type == "np.array":
            chunk_embedding = np.array(chunk_embedding).reshape(1, -1)
        return chunk_embedding

    def embed_question(self, question: str, output_type: str) -> np.ndarray:
        """
        Embed a question.
        """
        question_embedding = self._embed(question)
        if output_type == "np.array":
            question_embedding = np.array(question_embedding).reshape(1, -1)
        return question_embedding

    def _embed(self, content: str) -> list:
        raise NotImplementedError("Subclasses should implement this method.")


class LocalEmbedder(BaseEmbedder):
    """
    LocalEmbedder class for local SentenceTransformer model. Inherits from BaseEmbedder class.
    """

    def __init__(self):
        self.emb_model = self.load_local_model(settings.LOCAL_EMBEDDING_MODEL)

    @staticmethod
    def load_local_model(model_name):
        """
        Load the local SentenceTransformer model.
        """
        model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False},
        )
        return model

    def _embed(self, content: str) -> list:
        return self.emb_model.encode(content).tolist()


class RemoteEmbedder(BaseEmbedder):
    """
    RemoteEmbedder class for remote MistralEmbeddingAPI. Inherits from BaseEmbedder class.
    """

    def __init__(self):
        self.emb_api = llm_apis.MistralEmbeddingAPI()

    def _embed(self, content: str) -> list:
        return self.emb_api.call_api(content)["content"]


@click.command()
@click.option("--path_to_chunk_folder", type=click.Path(exists=True), help="Path to the chunked content folder.")
@click.option("--output_folder", type=click.Path(), help="Path to the output folder.")
@click.option("--type", type=str, help="Type of the embedding model (local or remote).")
def embed_chunks(path_to_chunk_folder, output_folder, type):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    document_name = path_to_chunk_folder.split("/")[-1]
    with open(os.path.join(path_to_chunk_folder, f"{document_name}.json"), "r") as file:
        chunks = json.load(file)

    if type == "local":
        embedder = LocalEmbedder()
    elif type == "remote":
        embedder = RemoteEmbedder()
    else:
        raise ValueError(f"Invalid type: {type}")

    embeddings = {}
    for chunk in chunks:
        embeddings[chunk["chunk_id"]] = {
            "type": chunk["type"],
            "page_class": chunk["page_class"],
            "embeddings": embedder.embed_chunk(chunk, output_type="list"),
        }

    output_file_path = os.path.join(output_folder, f"{document_name}.json")
    with open(output_file_path, "w") as f:
        json.dump(embeddings, f, indent=4)
