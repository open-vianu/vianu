import logging

logger = logging.getLogger("fraudcrawler_logger")


class Processor:
    """
    Processes the product data and applies specific filtering rules.
    """

    LOCATION_MAPPING = {
        "Switzerland": "ch",
        "Chile": "cl",
        "Austria": "at",
    }

    def __init__(self, location):
        """
        Initializes the Processor with the given country code.

        Args:
            location (str): The location associated to the specific user.
        """
        self.country_code = self.LOCATION_MAPPING[location].lower()

    def process(self, products):
        """
        Processes the product data and filters based on country code.

        Args:
            products (list): A list of product data dictionaries.

        Returns:
            filtered_products: A filtered list of product data dictionaries.
        """
        logger.info(
            f"Processing {len(products)} products and filtering by country code: {self.country_code.upper()}"
        )

        filtered_products = []
        for product in products:
            url = product.get("url", "")
            if (
                f".{self.country_code}/" in url.lower()
                or url.lower().endswith(f".{self.country_code}")
                or ".com" in url.lower()
            ):
                filtered_products.append(product)

        logger.info(
            f"Filtered down to {len(filtered_products)} products after applying country code filter."
        )
        return filtered_products
