
import pytest
from vianu.spock.src.chunking import TextChunking

def test_get_chunks_zero_remaining():
    min_chunk_size = 3
    min_chunk_overlap = 2
    N = 9
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4",
        "3 4 5 6 7",
        "6 7 8 9",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks


def test_get_chunks_one_remaining():
    min_chunk_size = 3
    min_chunk_overlap = 2
    N = 10
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4",
        "3 4 5 6 7 8",
        "7 8 9 10",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks


def test_get_chunks_two_remaining():
    min_chunk_size = 3
    min_chunk_overlap = 2
    N = 11
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5",
        "4 5 6 7 8",
        "7 8 9 10 11",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks


def test_get_chunks_one_chunk_exact():
    min_chunk_size = 10
    min_chunk_overlap = 2
    N = 10
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5 6 7 8 9 10",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks


def test_get_chunks_one_chunk_over():
    min_chunk_size = 11
    min_chunk_overlap = 2
    N = 10
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5 6 7 8 9 10",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks



def test_get_chunks_one_chunk_plus_overlap_sufficient():
    min_chunk_size = 9
    min_chunk_overlap = 2
    N = 10
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5 6 7 8 9 10",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks



def test_get_chunks_two_chunks():
    min_chunk_size = 7
    min_chunk_overlap = 2
    N = 10
    chunking = TextChunking(min_chunk_size=min_chunk_size, min_chunk_overlap=min_chunk_overlap)
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5 6 7 8 9 10",
    ]
    chunks = chunking.get_chunks(text)
    assert chunks == expected_chunks

