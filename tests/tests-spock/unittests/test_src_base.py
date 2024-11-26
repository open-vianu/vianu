from datetime import datetime
from pathlib import Path

import pytest

from vianu.spock.src import base

@pytest.fixture
def spock():
    date = datetime.strptime('2021-01-01', '%Y-%m-%d')
    setup = base.Setup(
        id_='id',
        log_level='INFO',
        max_docs_src=5,
        term='term',
        source=['source'],
        n_scp_tasks=1,
        model='model',
        n_ner_tasks=1,
        submission=date,
    )

    mp = base.NamedEntity(id_='id', text='text', class_='MP', location=(1, 5))
    data = [base.Document(id_='id', text='text', source='source', publication_date=date, medicinal_products=[mp])]

    spock = base.SpoCK(
        id_='id',
        status='status',
        started_at=date,
        finished_at=date, 
        setup=setup,
        data=data,
    )
    return spock

def test_serializable(spock: base.SpoCK):
    spock_dict = spock.to_dict()
    assert isinstance(spock_dict, dict)
    assert 'status' in spock_dict
    assert 'started_at' in spock_dict
    assert 'finished_at' in spock_dict
    assert 'setup' in spock_dict
    assert 'data' in spock_dict
    assert 'log_level' in spock_dict['setup']
    assert 'text' in spock_dict['data'][0]


def test_read_write(spock: base.SpoCK):
    file_path = Path(__file__).parents[1] / 'data'
    file_name = 'test.json'
    handler = base.FileHandler(file_path=file_path)
    handler.write(file_name=file_name, spock=spock, add_dt=False)
    spock_read = handler.read(file_name)
    assert spock == spock_read
    