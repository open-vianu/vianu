import unittest
import pandas as pd

from unittest.mock import MagicMock, patch
from vianu.fraudcrawler.src.enrichment import KeywordEnricher

class TestKeywordEnricher(unittest.TestCase):

    @patch("vianu.fraudcrawler.src.serpapi.SerpApiClient")
    @patch("vianu.fraudcrawler.src.enrichment.DataforSeoAPI")
    def test_apply_with_mocked_apis(self, MockDataforSeoAPI, MockSerpApiClient):
        """
        Test evaluated the KeywordEnricher.apply() method with mocked API responses.
        """

        # Mock SerpApiClient instance
        mock_serp_client = MockSerpApiClient.return_value
        mock_serp_client.call_serpapi.return_value = {"organic_results": [{"url": "https://example.com"}]}
        mock_serp_client.get_organic_results.return_value = [{"url": "https://example.com"}]
        mock_serp_client._check_limit.return_value = [{"url": "https://example.com", "score": 1}]

        # Mock DataforSeoAPI instance
        mock_dataforseo = MockDataforSeoAPI.return_value
        mock_dataforseo.get_keyword_suggestions.return_value = [
            {"keywordEnriched": "sildenafil online", "keywordVolume": 100, "keywordLocation": "Switzerland", "keywordLanguage": "en", "offerRoot": "SUGGESTED"}
        ]
        mock_dataforseo.get_related_keywords.return_value = [
            {"keywordEnriched": "buy sildenafil", "keywordVolume": 150, "keywordLocation": "Switzerland", "keywordLanguage": "en", "offerRoot": "RELATED"}
        ]

        # Mock enrichment utilities
        with patch("vianu.fraudcrawler.src.enrichment_utils.filter_keywords", return_value="filtered_keyword") as mock_filter:
            with patch("vianu.fraudcrawler.src.enrichment_utils.aggregate_keywords", return_value=pd.DataFrame([{
                "keywordEnriched": "sildenafil online",
                "keywordVolume": 100,
                "keywordLocation": "Switzerland",
                "keywordLanguage": "en",
                "offerRoot": "SUGGESTED"
            }])) as mock_aggregate:
                with patch("vianu.fraudcrawler.src.enrichment_utils.estimate_volume_per_url", return_value=[
                    {"url": "https://example.com", "traffic": 200, 'keywordVolume': 4356.0, 'keywordEnriched': 'sildenafil', 'keywordLanguage': 'German', 'keywordLocation': 'Switzerland', 'offerRoot': 'KEYWORD_SUGGESTION'}
                ]) as mock_estimate:

                    with patch.object(KeywordEnricher, "retrieve_response", return_value={"organic_results": [{"url": "https://example.com"}]}) as mock_retrieve_response:

                        # Initialize KeywordEnricher with mocked clients
                        enricher = KeywordEnricher(serpapi_key="fake_key", zyte_api_key="fake_key", location="Switzerland")

                        # Call the apply method
                        df = enricher.apply(keyword="sildenafil", number_of_keywords=5, language="en")

                        # Assertions
                        self.assertIsInstance(df, pd.DataFrame)  # Ensure return type is a DataFrame
                        self.assertEqual(len(df), 1)  # Expecting one result from mocked data
                        self.assertIn("url", df.columns)  # Ensure "link" column exists
                        self.assertEqual(df["url"].iloc[0], "https://example.com")  # Ensure correct mocked URL

                        # Verify mocks were called
                        mock_dataforseo.get_keyword_suggestions.assert_called_once_with("sildenafil", "Switzerland", "en", 5)
                        mock_dataforseo.get_related_keywords.assert_called_once_with("sildenafil", "Switzerland", "en", 5)
                        mock_filter.assert_called()
                        mock_aggregate.assert_called()
                        mock_estimate.assert_called()


if __name__ == "__main__":
    unittest.main()