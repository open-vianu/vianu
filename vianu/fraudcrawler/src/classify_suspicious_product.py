import openai
import logging
import time

from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ClassifySuspiciousProduct:
    """A client to classify products based on a user-defined context using OpenAI's GPT model."""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o", max_retries: int = 3, retry_delay: float = 2.0):
        """Initializes the ClassifyProductDomainClient with the given API key.

        Args:
            api_key: The API key for OpenAI API.
            model: The OpenAI model to use (default: "gpt-4o").
        """
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def classify_individual_suspicious_product(self, product: Dict[str, Any]) -> int:
        """Classifies a single product as relevant (1) or not relevant (0) based on the given context.

        Args:
            product: The product json contents including URL, PRODUCT_NAME, PRODUCT_DESCRIPTION.

        Returns:
            int: 1 if relevant, 0 if not relevant.
        """
        system_prompt = (
            "You are a helpful and intelligent assistant. Your task is to classify any given product "
            "as either relevant (1) or not relevant (0), strictly based on the context provided by the user. "
            "You must consider all aspects of the given context and make a binary decision accordingly. "
            "If the product aligns with the user's needs, classify it as 1 (relevant); otherwise, classify it as 0 (not relevant). "
            "Respond only with the number 1 or 0."
        )

        product = self.handle_missing_fields(product, 'name')
        product = self.handle_missing_fields(product, 'description')
        product_details = product['product']['name']+'\n'+product['product']['description']
        user_prompt = f"Product Details: {product_details}\nRelevance:"

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1  # Ensuring a short response
                )

                classification = response.choices[0].message.content.strip()

                if classification in ["0", "1"]:
                    logger.info(f"CLASSIFIED PRODUCT -- {product['product']['name']} -- as {classification}")
                    return int(classification)
                else:
                    raise ValueError(f"Unexpected response from OpenAI API: {classification}")

                return int(classification)

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)  # Wait before retrying
                else:
                    logger.error(f"Error classifying product after {self.max_retries} attempts: {e}")
                    return -1  # Indicate an error occurred
        return -1

    def classify_suspicious_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classifies multiple products based on the given context.

        Args:
            products: A list of product dictionaries.

        Returns:
            List[int]: A list of classification results (1 for relevant, 0 for not relevant).
        """
        for product in products:
            product["is_suspicious"] = self.classify_individual_suspicious_product(product)
        return products

    @staticmethod
    def handle_missing_fields(product: Dict[str, Any], field: str) -> Dict[str, Any]:
        try:
            product_name = product["product"][field] # noqa: F841
        except Exception as e:
            product["product"][field] = "MISSING DATA, VIANU MODIFIED - it is a relevant product"
            logger.error(f"PRODUCT DATA IN FIELD field does not exist. Error displayed: {e}")
        return product
