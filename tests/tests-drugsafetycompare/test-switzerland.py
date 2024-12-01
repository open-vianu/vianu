import pytest
from unittest.mock import patch, MagicMock
from vianu.drugsafetycompare.src.extract_switzerland import SwissDrugInfoExtractor


@pytest.fixture
def mock_webdriver():
    with patch(
        "vianu.drugsafetycompare.src.extract_switzerland.webdriver.Chrome"
    ) as mock:
        yield mock


def test_search_drug_success(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content with two products (using tr and td elements)
    sample_html = """
    <table class="table">
        <tbody>
            <tr data-url="/product/1">
                <td>Aspirin 100mg</td>
                <td>Some info</td>
                <td>Additional info</td>
            </tr>
            <tr data-url="/product/2">
                <td>Aspirin 500mg</td>
                <td>Some info</td>
                <td>Additional info</td>
            </tr>
        </tbody>
    </table>
    """
    mock_driver_instance.page_source = sample_html

    # Mock find_elements to return rows
    mock_row_1 = MagicMock()
    mock_row_1.find_elements.return_value = [
        MagicMock(text="Aspirin 100mg"),
        MagicMock(text="Info"),
        MagicMock(text="Info"),
    ]
    mock_row_1.get_attribute.return_value = "/product/1"
    mock_row_2 = MagicMock()
    mock_row_2.find_elements.return_value = [
        MagicMock(text="Aspirin 500mg"),
        MagicMock(text="Info"),
        MagicMock(text="Info"),
    ]
    mock_row_2.get_attribute.return_value = "/product/2"
    mock_driver_instance.find_elements.return_value = [mock_row_1, mock_row_2]

    # Initialize the extractor
    extractor = SwissDrugInfoExtractor()

    # Call the method
    products = extractor.search_drug("Aspirin")

    # Define expected output
    expected_products = [
        {"name": "Aspirin 100mg", "link": "https://sai.refdata.ch/product/1"},
        {"name": "Aspirin 500mg", "link": "https://sai.refdata.ch/product/2"},
    ]

    # Assert the results
    assert products == expected_products


def test_search_drug_no_products(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Sample HTML content with no products
    sample_html = """
    <table class="table">
        <tbody>
        </tbody>
    </table>
    """
    mock_driver_instance.page_source = sample_html

    # Mock find_elements to return no rows
    mock_driver_instance.find_elements.return_value = []

    # Initialize the extractor
    extractor = SwissDrugInfoExtractor()

    # Call the method
    products = extractor.search_drug("NonExistentDrug")

    # Define expected output
    expected_products = []

    # Assert the results
    assert products == expected_products


def test_get_side_effects_success(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Mock the 'Fachinformation' button click
    mock_fachinfo_button = MagicMock()
    mock_driver_instance.find_element.return_value = mock_fachinfo_button
    mock_fachinfo_button.click.return_value = None  # Simulate click

    # Mock window handles to simulate a new window opening
    mock_driver_instance.window_handles = ["window1", "window2"]

    # Sample shadow DOM content with 'Unerwünschte Wirkungen'
    shadow_dom_content = """
    <p>Unerwünschte Wirkungen</p>
    <p>Headache</p>
    <p>Nausea</p>
    """

    # Mock execute_script to return different values based on input script
    def mock_execute_script(script):
        if "shadowRoot != null" in script:
            return True
        elif "shadowRoot.innerHTML" in script:
            return shadow_dom_content
        return ""

    mock_driver_instance.execute_script.side_effect = mock_execute_script

    # Initialize the extractor
    extractor = SwissDrugInfoExtractor()

    # Call the method
    side_effects = extractor.get_side_effects("https://sai.refdata.ch/product/1")

    # Define expected output
    expected_side_effects = "Headache\nNausea"

    # Assert the results
    assert side_effects == expected_side_effects


def test_get_side_effects_no_fachinformation(mock_webdriver):
    # Mock the WebDriver instance and its methods
    mock_driver_instance = MagicMock()
    mock_webdriver.return_value = mock_driver_instance

    # Simulate no new window opening after clicking 'Fachinformation'
    mock_driver_instance.find_element.return_value = MagicMock()
    mock_driver_instance.window_handles = ["window1"]  # Only one window

    # Initialize the extractor
    extractor = SwissDrugInfoExtractor()

    # Call the method
    side_effects = extractor.get_side_effects("https://sai.refdata.ch/product/1")

    # Define expected output
    expected_output = "No new window opened after clicking 'Fachinformation'."

    # Assert the results
    assert side_effects == expected_output


def test_get_side_effects_exception_handling(mock_webdriver):
    # Mock the WebDriver instance and its methods to raise an exception during find_element
    mock_driver_instance = MagicMock()
    mock_driver_instance.find_element.side_effect = Exception("Element not found")
    mock_webdriver.return_value = mock_driver_instance

    # Initialize the extractor
    extractor = SwissDrugInfoExtractor()

    # Call the method
    side_effects = extractor.get_side_effects("https://sai.refdata.ch/product/1")

    # Define expected output
    expected_output = (
        "Could not find or click the 'Fachinformation' button: Element not found"
    )

    # Assert the results
    assert side_effects == expected_output
