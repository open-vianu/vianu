"""
Script to search for a drug and retrieve undesired effects of a specific product.

This script demonstrates how to use the GermanDrugInfoExtractor class from germanpy.py
to search for the drug "aspirin", select the third product in the search results,
and print its undesired effects (Nebenwirkungen).
"""

from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor


extractor = GermanDrugInfoExtractor()

try:
    # Define the drug name
    drug_name = "aspirin"

    # Search for products matching the drug name
    products = extractor.search_drug(drug_name)

    # Select the third product (index 2)
    selected_product = products[2]

    # Get the product URL
    product_url = selected_product["link"]

    # Retrieve undesired effects
    side_effects = extractor.get_undesired_effects(product_url)

    # Print the results
    print(f"Selected Product: {selected_product['name']}\n")
    print("Undesired Effects:")
    print(side_effects)

finally:
    # Close the extractor
    extractor.quit()
