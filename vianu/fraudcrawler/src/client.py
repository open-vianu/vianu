import json
import pandas as pd
import logging

from vianu.fraudcrawler.src.utils import transform_df_to_json
from vianu.fraudcrawler.src.processor import Processor
from vianu.fraudcrawler.src.serpapi import SerpApiClient
from vianu.fraudcrawler.src.zyteapi import ZyteAPIClient

logger = logging.getLogger("fraudcrawler_logger")


class FraudCrawlerClient:
    """
    The main client that orchestrates the search, data fetching, and processing.
    """

    def __init__(self, serpapi_token=None, zyte_api_key=None):
        """
        Initializes the FraudCrawlerClient with optional API tokens.

        Args:
            serpapi_token (str, optional): The API token for SERP API.
            zyte_api_key (str, optional): The API key for Zyte API.
        """
        self.serpapi_token = serpapi_token
        self.zyte_api_key = zyte_api_key

    def search(self, query, location, num_results=10):
        """
        Performs the search, gets product details, processes them, and returns a DataFrame.

        Args:
            query (str): The search query.
            location (str): The location of the user
            num_results (int): Number of search results to process.

        Returns:
            DataFrame: A pandas DataFrame containing the final product data.
        """
        # Ensure API tokens are set
        if not self.serpapi_token:
            raise ValueError("SERP API token is not set.")
        if not self.zyte_api_key:
            raise ValueError("Zyte API key is not set.")

        # Instantiate clients
        serp_client = SerpApiClient(self.serpapi_token, location)
        zyte_client = ZyteAPIClient(self.zyte_api_key)
        processor = Processor(location)

        # Perform search
        urls = serp_client.search(query, num_results)
        if not urls:
            logger.error("No URLs found from SERP API.")
            return pd.DataFrame()

        # Get product details
        products = zyte_client.get_details(urls)
        if not products:
            logger.error("No product details fetched from Zyte API.")
            return pd.DataFrame()

        # Process products
        filtered_products = processor.process(products)
        if not filtered_products:
            logger.warning("No products left after filtering.")
            return pd.DataFrame()

        # Flatten the product data
        df = pd.json_normalize(filtered_products)

        # Transform to Json
        query_info = {
            "search_term": query,
            "num_results": num_results,
            "location": location
        }
        print(len(df))
        products_json = transform_df_to_json(query_info, df)

        # Save to file
        with open("output.json", "w", encoding="utf-8") as f:
            json.dump(products_json, f, indent=4)
        logger.info("JSON file has been successfully created: output.json")

        # Log and return the DataFrame
        logger.info("Search completed. Returning flattened DataFrame.")
        return df, products_json
