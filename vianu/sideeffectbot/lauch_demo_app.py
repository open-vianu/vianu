"""
Gradio app for searching drug information and displaying undesired effects from Germany and Switzerland.

This script sets up a Gradio interface that allows users to input a drug name,
and view products and side effects from both Germany and Switzerland.
"""

import gradio as gr
from vianu.sideeffectbot.src.germany import GermanDrugInfoExtractor
from vianu.sideeffectbot.src.switzerland import SwissDrugInfoExtractor

def show_results_sections():
    """
    Makes the results sections visible immediately.
    """
    return gr.update(visible=True), gr.update(visible=True)

def search_and_display(drug_name):
    """
    Searches for products in both countries and updates the dropdowns.

    Args:
        drug_name (str): The name of the drug to search for.

    Returns:
        Tuple[gr.update]: Update objects for the German and Swiss Dropdown components.
    """
    # Search in Germany
    german_products = german_extractor.search_drug(drug_name)
    german_product_names = [product['name'] for product in german_products]
    global german_products_dict
    german_products_dict = {product['name']: product['link'] for product in german_products}

    # Search in Switzerland
    swiss_products = swiss_extractor.search_drug(drug_name)
    swiss_product_names = [product['name'] for product in swiss_products]
    global swiss_products_dict
    swiss_products_dict = {product['name']: product['link'] for product in swiss_products}

    # Prepare updates for the German section
    if german_product_names:
        german_dropdown_update = gr.update(choices=german_product_names, value=german_product_names[0])
        german_side_effects_output_update = gr.update(visible=True)
    else:
        german_dropdown_update = gr.update(choices=[], value=None)
        german_side_effects_output_update = gr.update(visible=False)

    # Prepare updates for the Swiss section
    if swiss_product_names:
        swiss_dropdown_update = gr.update(choices=swiss_product_names, value=swiss_product_names[0])
        swiss_side_effects_output_update = gr.update(visible=True)
    else:
        swiss_dropdown_update = gr.update(choices=[], value=None)
        swiss_side_effects_output_update = gr.update(visible=False)

    return (german_dropdown_update, german_side_effects_output_update,
            swiss_dropdown_update, swiss_side_effects_output_update)

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

# Initialize the extractors
german_extractor = GermanDrugInfoExtractor()
swiss_extractor = SwissDrugInfoExtractor()

# Global variables to store product information
german_products_dict = {}
swiss_products_dict = {}

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # Drug Information Search

    Enter the name of a drug to search for products and view their undesired effects (Nebenwirkungen).
    """)
    with gr.Row():
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        search_button = gr.Button("Search")

    # German Results Section (initially hidden)
    german_section = gr.Group(visible=False)
    with german_section:
        gr.Markdown("## Results for Germany")
        german_dropdown = gr.Dropdown(label="Select a Product (Germany)", choices=[])
        german_side_effects_output = gr.Textbox(label="Undesired Effects (Germany)", lines=10)

    # Swiss Results Section (initially hidden)
    swiss_section = gr.Group(visible=False)
    with swiss_section:
        gr.Markdown("## Results for Switzerland")
        swiss_dropdown = gr.Dropdown(label="Select a Product (Switzerland)", choices=[])
        swiss_side_effects_output = gr.Textbox(label="Undesired Effects (Switzerland)", lines=10)

    # Define button actions
    search_button.click(
        fn=show_results_sections,
        inputs=None,
        outputs=[german_section, swiss_section]
    ).then(
        fn=search_and_display,
        inputs=drug_input,
        outputs=[german_dropdown, german_side_effects_output,
                 swiss_dropdown, swiss_side_effects_output]
    )

    # When a product is selected in the German dropdown
    german_dropdown.change(
        fn=get_german_side_effects,
        inputs=german_dropdown,
        outputs=german_side_effects_output
    )

    # When a product is selected in the Swiss dropdown
    swiss_dropdown.change(
        fn=get_swiss_side_effects,
        inputs=swiss_dropdown,
        outputs=swiss_side_effects_output
    )

    gr.Markdown("""
    ### Instructions

    1. **Enter Drug Name**: Input the name of the drug you wish to search for.
    2. **Search**: Click the 'Search' button to retrieve matching products.
    3. **Select a Product**: Choose a product from the dropdown menus for each country.
    4. **View Undesired Effects**: The undesired effects will be displayed below each dropdown.
    """)

# Close the extractors when the app is stopped
def on_close():
    german_extractor.quit()
    swiss_extractor.quit()

demo.launch()