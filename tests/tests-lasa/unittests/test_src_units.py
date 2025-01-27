import pytest

from vianu.lasa.src.units import Product, AuthorizationUnit, SwissmedicAuthorization, FDAAuthorization
from vianu.lasa.src.units import AuthorizationFactory
from vianu.lasa.src.base import LASA, Match
from vianu.lasa.__main__ import process

_N_SMC_PRODUCTS = 5657
_N_FDA_PRODUCTS = 8075

@pytest.fixture
def smc_auth():
    return AuthorizationFactory.create('swissmedic')

@pytest.fixture
def fda_auth():
    return AuthorizationFactory.create('fda')

def test_smc_auth_from_file(smc_auth: SwissmedicAuthorization):
    assert isinstance(smc_auth, AuthorizationUnit)
    assert isinstance(smc_auth, SwissmedicAuthorization)
    assert all([isinstance(p, Product) for p in smc_auth.products])
    assert len(smc_auth.products) == _N_SMC_PRODUCTS
    assert smc_auth.source == 'swissmedic'

def test_smc_auth_products_to_df(smc_auth: SwissmedicAuthorization):
    df = smc_auth.get_products_as_df()
    assert df.shape == (_N_SMC_PRODUCTS, 4)
    assert 'name' in df.columns
    assert 'license_holder' in df.columns
    assert 'valid_until' in df.columns
    assert 'active_substance' in df.columns

def test_smc_auth_process(smc_auth: SwissmedicAuthorization):
    products = smc_auth.get_products_as_df()
    ref = 'CABAZITAXEL'

    lasa = LASA(ref=ref)   
    matches = process(lasa=lasa, names=products['name'])

    assert len(matches) == len(products)
    assert all([isinstance(r, Match) for r in matches])
    assert all([0 <= r.combined <= 1 for r in matches])

def test_fda_auth_from_file(fda_auth: FDAAuthorization):
    assert isinstance(fda_auth, AuthorizationUnit)
    assert isinstance(fda_auth, FDAAuthorization)
    assert all([isinstance(p, Product) for p in fda_auth.products])
    assert len(fda_auth.products) == _N_FDA_PRODUCTS
    assert fda_auth.source == 'fda'

def test_fda_auth_products_to_df(fda_auth: FDAAuthorization):
    df = fda_auth.get_products_as_df()
    assert df.shape == (_N_FDA_PRODUCTS, 4)
    assert 'name' in df.columns
    assert 'license_holder' in df.columns
    assert 'valid_until' in df.columns
    assert 'active_substance' in df.columns
