import logging
from typing import List

logger = logging.getLogger("fraudcrawler_logger")


class Processor:
    """Processes the product data and applies specific filtering rules."""

    LOCATION_MAPPING = {
        "Switzerland": "ch",
        "Chile": "cl",
        "Austria": "at",
    }

    def __init__(self, location: str):
        """Initializes the Processor with the given location.

        Args:
            location: The location used to process the products.
        """
        country_code = self.LOCATION_MAPPING[location].lower()
        if country_code is None:
            logger.warning(
                f'Location {location} not found in self._location_mapping (defaulting to "ch").'
            )
            country_code = "ch"
        self._country_code = country_code.lower()

    def _keep_product(self, product: dict) -> bool:
        """Determines whether to keep the product based on the coutry_code.

        Args:
            product: A product data dictionary.
        """
        url = product.get("url", "")
        return (
            f".{self._country_code}/" in url.lower()
            or url.lower().endswith(f".{self._country_code}")
            or ".com" in url.lower()
        )

    def _filter_products(self, products: List[dict]) -> List[dict]:
        """Filters the products based on the country_code.

        Args:
            products: A list of product data dictionaries.
        """
        logger.debug(
            f'Filtering {len(products)} products by country_code "{self._country_code.upper()}".'
        )
        filtered = [prod for prod in products if self._keep_product(prod)]
        logger.debug(
            f"Filtered down to {len(filtered)} products due to country code filter."
        )
        return filtered

    def _filter_zyte_probability(self, products: List[dict], threshold: float=0.1) -> List[dict]:
        """Filters the products based on the zyte probability.

        Args:
            products: A list of product data dictionaries.
        """
        for product in products:
            try:
                prob = product['product']['metadata']['probability']
                logger.info(f"This product IS KEPT: {product['url']} with Zyte probability: {prob}" if prob > threshold else f"This product is discarded: {product['url']} with Zyte probability: {prob}")
            except KeyError as e:
                logger.error(f"Missing key: {e} in product: {product}")

        try:
            prob_filtered = [elem for elem in products if elem['product']['metadata']['probability'] > threshold]
        except KeyError as e:
            logger.error(f"Missing key: {e} in one of the products")
            prob_filtered = []  # Fallback to an empty list in case of error

        logger.info(f'FILTER ZYTE PROBABILITY - Total results is: {len(prob_filtered)}')

        return prob_filtered

    def process(self, products: List[dict]) -> List[dict]:
        """Processes the product data and filters based on country code.

        Args:
            products: A list of product data dictionaries.
        """
        logger.info(
            f"PROCESSOR: Processing {len(products)} products and filtering by country code: {self._country_code.upper()}"
        )

        # Filter products based on country code
        filtered_country_code = self._filter_products(products)

        logger.info(
            f"PROCESSOR: Finished processing with {len(filtered_country_code)} products after applying country code filter."
        )

        # Filter products based on Zyte probability
        processed = self._filter_zyte_probability(filtered_country_code)
        logger.info(
            f"PROCESSOR: {len(processed)} products after filtering by Zyte probability."
        )

        return processed
