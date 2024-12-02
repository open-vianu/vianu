# tests/tests-drugsafetycompare/test-germany.py

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor

class TestGermanDrugInfoExtractor(unittest.TestCase):
    @patch('vianu.drugsafetycompare.src.germany.webdriver.Chrome')
    def test_search_drug_success(self, mock_webdriver):
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
        self.assertEqual(products, expected_products)

    @patch('vianu.drugsafetycompare.src.germany.webdriver.Chrome')
    def test_search_drug_no_products(self, mock_webdriver):
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
        self.assertEqual(products, expected_products)

    @patch('vianu.drugsafetycompare.src.germany.webdriver.Chrome')
    def test_get_undesired_effects_success(self, mock_webdriver):
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
        side_effects = extractor.get_undesired_effects("https://www.rote-liste.de/produkt/1")

        # Define expected output
        expected_side_effects = "Kopfschmerzen, Übelkeit"

        # Assert the results
        self.assertEqual(side_effects, expected_side_effects)

    @patch('vianu.drugsafetycompare.src.germany.webdriver.Chrome')
    def test_get_undesired_effects_no_section(self, mock_webdriver):
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
        side_effects = extractor.get_undesired_effects("https://www.rote-liste.de/produkt/1")

        # Define expected output
        expected_output = "No 'Nebenwirkungen' section found."

        # Assert the results
        self.assertEqual(side_effects, expected_output)

    @patch('vianu.drugsafetycompare.src.germany.webdriver.Chrome')
    def test_get_undesired_effects_exception_handling(self, mock_webdriver):
        # Mock the WebDriver instance and its methods to raise an exception when accessing page_source
        mock_driver_instance = MagicMock()
        type(mock_driver_instance).page_source = PropertyMock(side_effect=Exception("Element not found"))
        mock_webdriver.return_value = mock_driver_instance

        # Initialize the extractor
        extractor = GermanDrugInfoExtractor()

        # Call the method and expect an exception
        with self.assertRaises(Exception) as context:
            extractor.get_undesired_effects("https://www.rote-liste.de/produkt/1")

        self.assertIn("Element not found", str(context.exception))

if __name__ == '__main__':
    unittest.main()