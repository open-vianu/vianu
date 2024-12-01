import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from vianu.drugsafetycompare.src.extract_germany import GermanDrugInfoExtractor


@pytest.fixture
def mock_webdriver():
    with patch("vianu.drugsafetycompare.src.extract_germany.webdriver.Chrome") as mock:
        yield mock


def test_search_drug_success(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content with two products (using divs with specific classes)
    sample_html = """
    <div class="row py-3 px-4 ml-3 ml-lg-0">
        <a href="/produkt/1">Aspirin 100mg</a>
    </div>
    <div class="row py-3 px-4 ml-3 ml-lg-0">
        <a href="/produkt/2">Aspirin 500mg</a>
    </div>
    """
    mock_driver_instance.page_source = sample_html

    # Initialize the extractor
    extractor = GermanDrugInfoExtractor()

    # Call the method
    products = extractor.search_drug("Aspirin")

    # Define expected output
    expected_products = [
        {"name": "Aspirin 100mg", "link": "https://www.rote-liste.de/produkt/1"},
        {"name": "Aspirin 500mg", "link": "https://www.rote-liste.de/produkt/2"},
    ]

    # Assert the results
    assert products == expected_products


def test_search_drug_no_products(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content with no products
    sample_html = """
    <div class="no-products">No products found.</div>
    """
    mock_driver_instance.page_source = sample_html

    # Initialize the extractor
    extractor = GermanDrugInfoExtractor()

    # Call the method
    products = extractor.search_drug("NonExistentDrug")

    # Define expected output
    expected_products = []

    # Assert the results
    assert products == expected_products


def test_get_undesired_effects_success(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content with 'Nebenwirkungen' section as siblings
    sample_html = """
    <div id="nebenwirkungen"></div>
    <div class="product-detail--box-subtitle">
        Kopfschmerzen, Übelkeit
    </div>
    """
    mock_driver_instance.page_source = sample_html

    # Initialize the extractor
    extractor = GermanDrugInfoExtractor()

    # Call the method
    side_effects = extractor.get_undesired_effects(
        "https://www.rote-liste.de/produkt/1"
    )

    # Define expected output
    expected_side_effects = "Kopfschmerzen, Übelkeit"

    # Assert the results
    assert side_effects == expected_side_effects


def test_get_undesired_effects_no_section(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content without 'Nebenwirkungen' section
    sample_html = """
    <div id="indikationen">
        <div class="product-detail--box-subtitle">
            Indikationen content here.
        </div>
    </div>
    """
    mock_driver_instance.page_source = sample_html

    # Initialize the extractor
    extractor = GermanDrugInfoExtractor()

    # Call the method
    side_effects = extractor.get_undesired_effects(
        "https://www.rote-liste.de/produkt/1"
    )

    # Define expected output
    expected_output = "No 'Nebenwirkungen' section found."

    # Assert the results
    assert side_effects == expected_output


def test_get_undesired_effects_exception_handling(mock_webdriver):
    # Mock the WebDriver instance and its methods to raise an exception when accessing page_source
    mock_driver_instance = MagicMock()
    type(mock_driver_instance).page_source = PropertyMock(
        side_effect=Exception("Element not found")
    )
    mock_webdriver.return_value = mock_driver_instance

    # Initialize the extractor
    extractor = GermanDrugInfoExtractor()

    # Call the method and expect an exception
    with pytest.raises(Exception) as excinfo:
        extractor.get_undesired_effects("https://www.rote-liste.de/produkt/1")

    assert "Element not found" in str(excinfo.value)
