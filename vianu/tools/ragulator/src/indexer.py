import json
import os
from typing import Dict, List

import click
import faiss
import numpy as np

import settings

# get current working directory inside the package
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Indexer:
    def __init__(self, index_path, dimension=1024):
        self.index_path = index_path
        self.dimension = dimension
        self.indexes = self.load_index()

    def load_index(self) -> None:
        # check if the index file exists
        if not os.path.exists(self.index_path):
            indexes = {}
            # Initialize a FAISS index for this particular set of embeddings
            index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
            # Store the index in the class property `indexes`
            indexes["chunks"] = index
            return indexes
        index = faiss.deserialize_index(np.load(self.index_path))
        indexes = {"chunks": index}
        return indexes

    @staticmethod
    def remove_not_allowed_content(embeddings_dict):
        """
        Remove not allowed content
        """
        return {
            k: v
            for k, v in embeddings_dict.items()
            if v["embeddings"] and v["page_class"] not in settings.NOT_ALLOWED_CLASSES
        }

    def build_index(self, embeddings_dict: Dict[str, List]) -> None:
        """
        Build the FAISS index for the given embeddings.
        """
        embeddings_dict = {k: v["embeddings"] for k, v in embeddings_dict.items()}
        ids = np.array(list(embeddings_dict.keys()))
        embeddings = np.array(list(embeddings_dict.values()))
        # Add embeddings to the FAISS index along with the corresponding IDs
        print(f"Building chunks index ")
        print(f"embedding shape: {embeddings.shape}")
        print(f"ids shape: {ids.shape}")
        index = self.indexes["chunks"]
        index.add_with_ids(embeddings, ids)
        # Store the index in the class property `indexes`
        self.indexes["chunks"] = index

    def save_index(self) -> None:
        index = faiss.serialize_index(self.indexes["chunks"])
        np.save(self.index_path, index)


@click.command()
@click.option("--path_to_embedding_folder", type=click.Path(exists=True), help="Path to the embedding folder.")
@click.option("--output_folder", type=click.Path(), help="Path to the output folder.")
@click.option("--type", type=str, help="Type of the system associated to the index.")
def update_index(path_to_embedding_folder, output_folder, type):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    document_name = path_to_embedding_folder.split("/")[-1]
    with open(os.path.join(path_to_embedding_folder, f"{document_name}.json"), "r") as file:
        embeddings_dict = json.load(file)

    local_embedder = Indexer(index_path=os.path.join(output_folder, f"{settings.INDEX_FILENAME}_{type}.npy"))
    embeddings_dict = local_embedder.remove_not_allowed_content(embeddings_dict)
    local_embedder.build_index(embeddings_dict)
    local_embedder.save_index()
