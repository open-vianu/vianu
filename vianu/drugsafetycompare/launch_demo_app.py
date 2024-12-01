"""
Gradio app for searching drug information and displaying undesired effects from Germany and Switzerland.

This script sets up a Gradio interface that allows users to input a drug name,
and view products and side effects from both Germany and Switzerland.
"""

import os
import logging
import gradio as gr
from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor
from vianu.drugsafetycompare.src.switzerland import SwissDrugInfoExtractor
from vianu.drugsafetycompare.src.compare import compare_drugs_with_gpt

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("fraudcrawler_logger")


def show_results_sections():
    """
    Makes the results sections visible immediately.
    """
    return gr.update(visible=True), gr.update(visible=True)


def show_comparison_section():
    """
    Makes the comparison sections visible immediately.
    """
    return gr.update(visible=True), gr.update(visible=True)


def search_and_display(drug_name):
    """
    Searches for products in both countries and updates the dropdowns and side effects.

    Args:
        drug_name (str): The name of the drug to search for.

    Returns:
        Tuple: Updates for dropdowns and side effects.
    """
    # Search in Germany
    german_products = german_extractor.search_drug(drug_name)
    german_product_names = [product["name"] for product in german_products]
    global german_products_dict
    german_products_dict = {
        product["name"]: product["link"] for product in german_products
    }

    # Search in Switzerland
    swiss_products = swiss_extractor.search_drug(drug_name)
    swiss_product_names = [product["name"] for product in swiss_products]
    global swiss_products_dict
    swiss_products_dict = {
        product["name"]: product["link"] for product in swiss_products
    }

    # Prepare German section updates
    if german_product_names:
        first_german_product = german_product_names[0]
        german_dropdown_update = gr.update(
            choices=german_product_names, value=first_german_product
        )
        german_side_effects = get_german_side_effects(first_german_product)
        german_link = german_products_dict[first_german_product]
        german_side_effects_output_update = gr.update(
            value=german_side_effects, visible=True
        )
        german_link_update = gr.update(
            value=f"<a href='{german_link}' target='_blank'>{german_link}</a>",
            visible=True,
        )
    else:
        german_dropdown_update = gr.update(choices=[], value=None)
        german_side_effects_output_update = gr.update(value="", visible=False)
        german_link_update = gr.update(value="", visible=False)

    # Prepare Swiss section updates
    if swiss_product_names:
        first_swiss_product = swiss_product_names[0]
        swiss_dropdown_update = gr.update(
            choices=swiss_product_names, value=first_swiss_product
        )
        swiss_side_effects = get_swiss_side_effects(first_swiss_product)
        swiss_link = swiss_products_dict[first_swiss_product]
        swiss_side_effects_output_update = gr.update(
            value=swiss_side_effects, visible=True
        )
        swiss_link_update = gr.update(
            value=f"<a href='{swiss_link}' target='_blank'>{swiss_link}</a>",
            visible=True,
        )
    else:
        swiss_dropdown_update = gr.update(choices=[], value=None)
        swiss_side_effects_output_update = gr.update(value="", visible=False)
        swiss_link_update = gr.update(value="", visible=False)

    return (
        german_dropdown_update,
        german_side_effects_output_update,
        german_link_update,
        swiss_dropdown_update,
        swiss_side_effects_output_update,
        swiss_link_update,
    )


def get_german_side_effects(selected_product_name):
    """
    Retrieves the undesired effects of the selected German product.

    Args:
        selected_product_name (str): The name of the selected product.

    Returns:
        str: The undesired effects text.
    """
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        side_effects = german_extractor.get_undesired_effects(product_url)
        return side_effects
    else:
        return "Product not found."


def get_swiss_side_effects(selected_product_name):
    """
    Retrieves the undesired effects of the selected Swiss product.

    Args:
        selected_product_name (str): The name of the selected product.

    Returns:
        str: The undesired effects text.
    """
    if selected_product_name in swiss_products_dict:
        product_url = swiss_products_dict[selected_product_name]
        side_effects = swiss_extractor.get_side_effects(product_url)
        return side_effects
    else:
        return "Product not found."


def update_german_link(selected_product_name):
    """
    Updates the German product link dynamically.

    Args:
        selected_product_name (str): The name of the selected product.

    Returns:
        str: The HTML link for the selected product.
    """
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""


def update_swiss_link(selected_product_name):
    """
    Updates the Swiss product link dynamically.

    Args:
        selected_product_name (str): The name of the selected product.

    Returns:
        str: The HTML link for the selected product.
    """
    if selected_product_name in swiss_products_dict:
        product_url = swiss_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""


def compare_drugs(token, drug_name, german_product, swiss_product):
    """
    Compare the drugs using GPT and return the Markdown table.

    Args:
        token (str): OpenAI API token.
        drug_name (str): The drug name.
        german_product (str): Selected product name from Germany.
        swiss_product (str): Selected product name from Switzerland.

    Returns:
        str: Markdown table of the comparison.
    """
    # Fetch the side effects descriptions for the selected products
    german_description = get_german_side_effects(german_product)
    swiss_description = get_swiss_side_effects(swiss_product)

    # Call GPT for comparison
    if not token.strip():
        return "OpenAI API token is missing. Please provide a valid token."

    comparison_result = compare_drugs_with_gpt(
        token, drug_name, german_description, swiss_description
    )

    # Ensure the result is a string
    if isinstance(comparison_result, str):
        return comparison_result
    else:
        return str(comparison_result)


# Initialize the extractors
german_extractor = GermanDrugInfoExtractor()
swiss_extractor = SwissDrugInfoExtractor()

# Global variables to store product information
german_products_dict = {}
swiss_products_dict = {}

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.HTML(
        "<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>DrugSafetyCompare</h1>"
    )

    with gr.Row():
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        default_token = os.getenv("OPENAI_TOKEN", "")
        token_input = gr.Textbox(
            label="OpenAI API Token (comparison only)",
            placeholder="Paste your OpenAI API token here",
            type="password",
            value=default_token,
        )

        search_button = gr.Button("Search")
        compare_button = gr.Button("Compare")

    # Comparison Section
    comparison_section = gr.Group(visible=False)
    with comparison_section:
        gr.HTML("<h2 style='text-align: center;'>Drug Comparison</h2>")
        comparison_output = gr.Markdown()

    # German Results Section
    german_section = gr.Group(visible=False)
    with german_section:
        gr.HTML("<h3 style='text-align: left;'>Results for Germany</h3>")
        german_dropdown = gr.Dropdown(label="Select a Product (Germany)", choices=[])
        german_side_effects_output = gr.Textbox(
            label="Undesired Effects (Germany)", lines=10
        )
        german_link_output = gr.HTML()

    # Swiss Results Section
    swiss_section = gr.Group(visible=False)
    with swiss_section:
        gr.HTML("<h3 style='text-align: left;'>Results for Switzerland</h3>")
        swiss_dropdown = gr.Dropdown(label="Select a Product (Switzerland)", choices=[])
        swiss_side_effects_output = gr.Textbox(
            label="Undesired Effects (Switzerland)", lines=10
        )
        swiss_link_output = gr.HTML()

    # Search button logic
    search_button.click(
        fn=show_results_sections, inputs=None, outputs=[german_section, swiss_section]
    ).then(
        fn=search_and_display,
        inputs=drug_input,
        outputs=[
            german_dropdown,
            german_side_effects_output,
            german_link_output,
            swiss_dropdown,
            swiss_side_effects_output,
            swiss_link_output,
        ],
    )

    # Compare button logic
    compare_button.click(
        fn=lambda: gr.update(visible=True), inputs=None, outputs=comparison_section
    ).then(
        fn=compare_drugs,
        inputs=[token_input, drug_input, german_dropdown, swiss_dropdown],
        outputs=comparison_output,
    )

    # Dropdown change logic for German and Swiss results
    german_dropdown.change(
        fn=lambda selected_product: (
            get_german_side_effects(selected_product),
            update_german_link(selected_product),
        ),
        inputs=german_dropdown,
        outputs=[german_side_effects_output, german_link_output],
    )

    swiss_dropdown.change(
        fn=lambda selected_product: (
            get_swiss_side_effects(selected_product),
            update_swiss_link(selected_product),
        ),
        inputs=swiss_dropdown,
        outputs=[swiss_side_effects_output, swiss_link_output],
    )

    gr.HTML("""
    <div style="text-align: left; font-size: 1em; margin-top: 1em;">
        <p><b>Instructions:</b></p>
        <ol>
            <li>Enter the name of the drug in the search field.</li>
            <li>Click <b>Search</b> to retrieve products from Germany and Switzerland.</li>
            <li>Select a product from the dropdowns to view its details and link.</li>
            <li>Provide your OpenAI token to enable drug comparison and click <b>Compare</b>.</li>
        </ol>
    </div>
    """)


# Close the extractors when the app is stopped
def on_close():
    german_extractor.quit()
    swiss_extractor.quit()


def main():
    demo.launch(
        debug=True
    )  # Add share=True if you want to create a 72h lasting demo deployment.


# Launch the app
if __name__ == "__main__":
    main()
