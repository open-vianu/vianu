from typing import Dict, List, Optional, Tuple

import numpy as np

import embedder # import LocalEmbedder, RemoteEmbedder
import indexer # import Indexer


class Retriever:
    def __init__(self, corpus, index_path, type):
        self.corpus = corpus
        self.index_path = index_path
        self.type = type
        if self.type == "local":
            self.embedder = LocalEmbedder()
        elif self.type == "remote":
            self.embedder = RemoteEmbedder()
        else:
            raise ValueError(f"Invalid type: {self.type}. Please choose from 'local' or 'remote'.")
        self.indexer = Indexer(index_path=index_path)
        self.indexes = self.indexer.indexes

    def _retrieve_top_k(
            self, q_embedding: np.ndarray, top_k: int = 5, index_names: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[int, float]]]:
        if not self.indexes:
            raise ValueError("No indexes found. Please build the indexes first using `Indexer` class.")

        if index_names is None:
            index_names = list(self.indexes.keys())

        results = {}

        for index_name in index_names:
            if index_name not in self.indexes:
                print(f"Warning: Index '{index_name}' not found. Skipping.")
                continue

            D, I = self.indexes[index_name].search(q_embedding, top_k)
            results[index_name] = [(int(id), float(dist)) for id, dist in zip(I[0], D[0]) if id != -1]

        return results

    def retrieve(
            self, question: str, top_k: int = 5, index_names: Optional[List[str]] = None, only_ids: bool = False
    ) -> Dict[str, List[Tuple[int, float]]]:
        # Get the embedding for the question
        question_embedding = self.embedder.embed_question(question, output_type="np.array")
        # Retrieve the top k results from the indexes
        retrieved_results = self._retrieve_top_k(question_embedding, top_k, index_names)
        retrieved_chunks = []
        for index_name, results in retrieved_results.items():
            for chunk_id, dist in results:
                chunk_id = str(chunk_id)
                if only_ids:
                    chunk_meta = {
                        "chunk_id": chunk_id,
                    }
                else:
                    chunk_meta = self.corpus[chunk_id].copy()

                chunk_meta["_index_name"] = index_name
                chunk_meta["_dist"] = dist
                retrieved_chunks.append(chunk_meta)

        return retrieved_chunks
