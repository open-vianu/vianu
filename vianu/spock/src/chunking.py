from argparse import ArgumentParser, Namespace
import logging
from typing import List

import numpy as np

from .data_model import Document, FileHandler, TextEntity
from . import utils as utl


MODULE_NAME = 'chunking'

class TextChunking:

    _separator = ' '

    def __init__(self, min_chunk_size: int, min_chunk_overlap: int):
        self._min_chunk_size = min_chunk_size
        self._min_chunk_overlap = min_chunk_overlap

    def get_chunks(self, text: str):
        """A given text is split into a number of entities which are used to generate overlapping subtexts (chunks).
        """
        entities = text.split(self._separator)
        N = len(entities)
        s = min(self._min_chunk_size, N)
        n = N // s
        d = (self._min_chunk_overlap+1) // 2
        
        bnd = [round(i) for i in np.linspace(0, 1, n+1) * N]

        chunks = []
        for dwn, up in zip(bnd[:-1], bnd[1:]):
            start = max(0, dwn-d)
            stop = min(N, up+d)
            chunks.append(self._separator.join(entities[start:stop]))
            if stop >= N:
                break

        return chunks

def cli_args():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    group.add_argument('--min-chunk-size', dest='min_chunk_size', type=int, required=True)
    group.add_argument('--min-chunk-overlap', dest='min_chunk_overlap', type=int, required=True)
    return parser


def _postprocess_text_entities(data: List[Document]) -> None:
    for doc in data:
        for te in doc.text_entities:
            loc = utl.get_loc_of_subtext(text=doc.get_raw_text(), sutext=te.text)
            te.location = loc


def apply(args_: Namespace, data: List[Document] | None = None, save_data: bool = True) -> None:
    min_chunk_size = args_.min_chunk_size
    min_chunk_overlap = args_.min_chunk_overlap

    if data is None:
        data = FileHandler(args_.data_load).read()

    logging.info(f'chunking the raw text of {len(data)} documents')

    chunker = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    for doc in data:
        text = doc.get_raw_text()
        if text is None:
            err_msg = f"no raw text found for document with id {doc.id_}"
            logging.warning(err_msg)
            doc.add_error(err_msg)
            continue
        chunks = chunker.get_chunks(text=text)
        logging.debug(f'split into {len(chunks)} chunks')
        for text in chunks:
            te = TextEntity(
                id_=text,
                text=text,
            )
            doc.add_text_entity(text_entity=te)
    
    logging.info('add locations of text entities')
    _postprocess_text_entities(data=data)
    
    if save_data:
        FileHandler(args_.data_dump).write(data)
