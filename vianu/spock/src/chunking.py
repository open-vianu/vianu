from argparse import ArgumentParser
import logging
from typing import List

import numpy as np


MODULE_NAME = 'chunker'

class TextChunker:

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
        d = max(self._min_chunk_overlap, (N - s*n) // n) // 2 + 1
        
        bnd = [round(i) for i in np.linspace(0, 1, n+2) * N]

        chunks = []
        for dwn, up in zip(bnd[:-1], bnd[1:]):
            start = max(0, dwn-d)
            stop = min(N, up+d)
            chunks.append(self._separator.join(entities[start:stop]))

        return chunks

def add_parser(subparser: ArgumentParser, parents: List[ArgumentParser]):
    parser = subparser.add_parser(MODULE_NAME, parents=parents)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', dest='text', type=str)
    group.add_argument('--text-file', dest='text_file')

    parser.add_argument('--min-chunk-size', dest='min_chunk_size', type=int, required=True)
    parser.add_argument('--min-chunk-overlap', dest='min_chunk_overlap', type=int, required=True)


def apply(args_):
    if args_.text:
        text = args_.text
    else:
        with open(args_.text_file, 'r') as file:
            text = file.read()
    min_chunk_size = args_.min_chunk_size
    min_chunk_overlap = args_.min_chunk_overlap

    logging.info(f'chunking text of length {len(text)}')
    chunker = TextChunker(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    chunks = chunker.get_chunks(text=text)
    logging.info(f'split into {len(chunks)} chunks')
    for c in chunks:
        logging.debug(c)
