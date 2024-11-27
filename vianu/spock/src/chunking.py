from argparse import ArgumentParser
import logging
from typing import List

import numpy as np


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
        d = max(self._min_chunk_overlap, (N - s*n) // n) // 2 + 1
        
        bnd = [round(i) for i in np.linspace(0, 1, n+2) * N]

        chunks = []
        for dwn, up in zip(bnd[:-1], bnd[1:]):
            start = max(0, dwn-d)
            stop = min(N, up+d)
            chunks.append(self._separator.join(entities[start:stop]))

        return chunks

def cli_args():
    parser = ArgumentParser(add_help=False)
    group = parser.add_argument_group(MODULE_NAME)
    txt_grp = group.add_mutually_exclusive_group(required=True)
    txt_grp.add_argument('--text', dest='text', type=str)
    txt_grp.add_argument('--text-file', dest='text_file')

    group.add_argument('--min-chunk-size', dest='min_chunk_size', type=int, required=True)
    group.add_argument('--min-chunk-overlap', dest='min_chunk_overlap', type=int, required=True)
    return parser


def apply(args_):
    if args_.text:
        text = args_.text
    else:
        with open(args_.text_file, 'r') as file:
            text = file.read()
    min_chunk_size = args_.min_chunk_size
    min_chunk_overlap = args_.min_chunk_overlap

    logging.info(f'chunking text of length {len(text)}')
    chunker = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    chunks = chunker.get_chunks(text=text)
    logging.info(f'split into {len(chunks)} chunks')
