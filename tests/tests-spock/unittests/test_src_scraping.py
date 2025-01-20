from vianu.spock.src import scraping as scp


def test_split_text_into_chunks_zero_remaning():
    chunk_size = 3
    N = 9
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3",
        "4 5 6",
        "7 8 9",
    ]
    chunks = scp.Scraper.split_text_into_chunks(text=text, chunk_size=chunk_size)
    assert chunks == expected_chunks


def test_get_chunks_one_remaining():
    chunk_size = 3
    N = 10
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3",
        "4 5 6 7",
        "8 9 10",
    ]
    chunks = scp.Scraper.split_text_into_chunks(text=text, chunk_size=chunk_size)
    print(chunks)
    assert chunks == expected_chunks


def test_get_chunks_two_remaining():
    chunk_size = 3
    N = 11
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4",
        "5 6 7",
        "8 9 10 11",
    ]
    chunks = scp.Scraper.split_text_into_chunks(text=text, chunk_size=chunk_size)
    assert chunks == expected_chunks


def test_get_chunks_one_chunk_exact():
    chunk_size = 10
    N = 10
    text = ' '.join([str(i+1) for i in range(N)])

    expected_chunks = [
        "1 2 3 4 5 6 7 8 9 10",
    ]
    chunks = scp.Scraper.split_text_into_chunks(text=text, chunk_size=chunk_size)
    assert chunks == expected_chunks
