"""
Gradio App for Drug Information Scraping and Comparison.

This application allows users to:
1. Search for a drug and retrieve its products from Germany and Switzerland.
2. Select specific products to view their side effects.
3. Compare the side effects using zero-shot classification, radar charts, and SHAP explanations.
"""

import logging
import gradio as gr
import torch
from transformers import pipeline
import plotly.graph_objects as go
import shap
import numpy as np
import html
import re
import sys
import os
import atexit


# --------------------- Add 'src' to Python Path ---------------------
# Adjust the path according to your project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Import the necessary extractors from App 1
from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor
from vianu.drugsafetycompare.src.switzerland import SwissDrugInfoExtractor

# Initialize Logger
logger = logging.getLogger("drug_compare_logger")
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
# Add console handler for logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------- Initialize Extractors ---------------------
german_extractor = GermanDrugInfoExtractor()
swiss_extractor = SwissDrugInfoExtractor()

# --------------------- Global Product Dictionaries ---------------------
german_products_dict = {}
swiss_products_dict = {}

# --------------------- GPU Utilization ---------------------
# Determine if CUDA (GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.debug("Using device: MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.debug("Using device: cuda")
else:
    device = torch.device("cpu")
    logger.debug("Using device: cpu")

# Initialize the zero-shot classifier with GPU support if available
try:
    classifier = pipeline(
        "zero-shot-classification", model="facebook/bart-large-mnli", device=device
    )
    logger.debug("Zero-shot classifier initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing zero-shot classifier: {e}")
    classifier = None

# --------------------- Define SOCs ---------------------
# Define SOCs based on MedDRA
socs = [
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Congenital, familial and genetic disorders",
    "Endocrine disorders",
    "Eye disorders",
    "Gastrointestinal disorders",
    "General disorders and administration site conditions",
    "Hepatobiliary disorders",
    "Immune system disorders",
    "Infections and infestations",
    "Injury, poisoning and procedural complications",
    "Investigations",
    "Metabolism and nutrition disorders",
    "Musculoskeletal and connective tissue disorders",
    "Neoplasms benign, malignant and unspecified",
    "Nervous system disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Reproductive system and breast disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Skin and subcutaneous tissue disorders",
    "Social circumstances",
    "Surgical and medical procedures",
    "Vascular disorders",
]


# --------------------- Classification Function ---------------------
def classify_adverse_events(text, candidate_labels):
    """
    Classify the given text into multiple candidate labels using zero-shot classification.
    Returns a dictionary of SOCs with their corresponding normalized scores.
    """
    if not text.strip():
        # If the text is empty, return zero scores for all SOCs
        return {soc: 0.0 for soc in candidate_labels}

    if classifier is None:
        logger.error("Zero-shot classifier is not initialized.")
        return {soc: 0.0 for soc in candidate_labels}

    try:
        result = classifier(text, candidate_labels, multi_label=True)
        scores = dict(zip(result["labels"], result["scores"]))
        # Normalize scores to range [0, 1]
        max_score = max(scores.values()) if scores else 1
        normalized_scores = {soc: score / max_score for soc, score in scores.items()}
        logger.debug(f"Classification scores: {normalized_scores}")
        return normalized_scores
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return {soc: 0.0 for soc in candidate_labels}


# --------------------- Radar Chart Plotting Function ---------------------
def plot_radar_chart_plotly(socs, scores_a, scores_b, selected_soc=None):
    """
    Plots a radar chart using Plotly comparing two sets of scores across multiple categories.
    Optionally highlights a selected SOC.
    Returns the Plotly figure.
    """
    categories = socs.copy()
    categories += [socs[0]]  # Complete the loop for radar chart

    values_a = [scores_a.get(soc, 0) for soc in socs]
    values_a += [scores_a.get(socs[0], 0)]
    values_b = [scores_b.get(soc, 0) for soc in socs]
    values_b += [scores_b.get(socs[0], 0)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_a,
            theta=categories,
            fill="toself",
            name="Germany",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=values_b,
            theta=categories,
            fill="toself",
            name="Switzerland",
            line=dict(color="red"),
        )
    )

    # Highlight selected SOC
    if selected_soc and selected_soc in socs:
        fig.add_trace(
            go.Scatterpolar(
                r=[scores_a.get(selected_soc, 0), scores_b.get(selected_soc, 0)],
                theta=[selected_soc, selected_soc],
                mode="markers",
                marker=dict(size=12, color="green"),
                name=f"Highlighted SOC: {selected_soc}",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(scores_a.values()), max(scores_b.values()), 0.6)],
            ),
            angularaxis=dict(
                # Prevent labels from being cut off by increasing the padding
                rotation=90,
                direction="clockwise",
                showline=True,
                showticklabels=True,
                tickfont=dict(size=10),
                linewidth=1,
                tickangle=0,
            ),
        ),
        margin=dict(
            l=100, r=100, t=100, b=100
        ),  # Increased margins to prevent label cutoff
        showlegend=True,
        title="Comparison of Drug Toxicity Profiles by SOC",
        height=700,
        width=700,
        # Enable panning and zooming
        dragmode="pan",
    )

    return fig


# --------------------- SHAP Explainer Initialization ---------------------
def soc_model(texts, soc, *args, **kwargs):
    """
    Model function for SHAP that returns scores for a specific SOC.
    Accepts arbitrary additional arguments to prevent errors.
    """
    logger.debug(
        f"soc_model called with texts: {texts}, soc: {soc}, args: {args}, kwargs: {kwargs}"
    )
    if isinstance(texts, str):
        texts = [texts]
    scores = []
    for txt in texts:
        try:
            result = classifier(txt, [soc], multi_label=True)
            if isinstance(result, dict):
                # Single input
                score = result["scores"][0] if result["scores"] else 0.0
            elif isinstance(result, list) and len(result) > 0:
                # Multiple inputs
                score = result[0]["scores"][0] if result[0]["scores"] else 0.0
            else:
                score = 0.0
            scores.append(score)
        except Exception as e:
            logger.error(f"Error during classification in soc_model: {e}")
            scores.append(0.0)
    return np.array(scores)  # Ensure a NumPy array is returned


# Initialize SHAP Explainer per SOC and cache them
shap_explainers = {}


def create_shap_explainer(soc):
    """
    Creates or retrieves a SHAP explainer for the specified SOC.
    """
    if soc not in shap_explainers:
        try:
            # Use lambda to fix 'soc' parameter
            model_fn = lambda texts: soc_model(texts, soc)
            shap_explainer = shap.Explainer(model_fn, masker=shap.maskers.Text())
            shap_explainers[soc] = shap_explainer
            logger.debug(f"SHAP explainer created for SOC: {soc}")
        except Exception as e:
            logger.error(f"Error creating SHAP explainer for SOC '{soc}': {e}")
            return None
    return shap_explainers[soc]


# --------------------- SHAP Explanation Function ---------------------
def explain_soc(shap_explainer, text):
    """
    Generates SHAP explanations for the given text using the provided SHAP explainer.
    Returns the explanation as an HTML formatted string with highlighted words.
    """
    # Check if text is empty
    if not text.strip():
        return "No side effect descriptions available for analysis."

    # Generate SHAP values for the input text
    try:
        shap_values = shap_explainer([text])[0]
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        return "Error generating SHAP explanations."

    # Get the words and corresponding SHAP values
    words = shap_values.data
    shap_values_list = shap_values.values  # SHAP values for each word

    if not words.size or not shap_values_list.size:
        return "No side effect descriptions available for analysis."

    # Handle tokenization: join subwords if necessary
    combined_words = []
    combined_shap = []
    current_word = ""
    current_shap = 0.0
    for word, shap_val in zip(words, shap_values_list):
        if word.startswith("##"):
            current_word += word[2:]
            current_shap += shap_val
        else:
            if current_word:
                combined_words.append(current_word)
                combined_shap.append(current_shap)
            current_word = word
            current_shap = shap_val
    if current_word:
        combined_words.append(current_word)
        combined_shap.append(current_shap)

    if not combined_words or not combined_shap:
        return "No significant words found for SHAP explanations."

    # Normalize SHAP values for coloring
    max_shap = max(abs(val) for val in combined_shap) if combined_shap else 1.0
    max_shap = max_shap if max_shap != 0 else 1.0  # Prevent division by zero

    # Define colors
    def get_color(val):
        """
        Maps SHAP value to a color.
        Positive values are green, negative values are red, neutral is light gray.
        The intensity is proportional to the magnitude of the SHAP value.
        """
        if val > 0.1:
            # Strong positive
            return f"rgba(0, 255, 0, {min(val / max_shap, 1)})"  # Green
        elif val < -0.1:
            # Strong negative
            return f"rgba(255, 0, 0, {min(-val / max_shap, 1)})"  # Red
        else:
            # Neutral
            return "rgba(211, 211, 211, 0.5)"  # Light gray

    # Escape HTML characters in words
    escaped_words = [html.escape(word) for word in combined_words]

    # Build the HTML string with highlighted words
    highlighted_text = ""
    for word, shap_val in zip(escaped_words, combined_shap):
        color = get_color(shap_val)
        # Optionally, set a minimum opacity for better visibility
        if abs(shap_val) < 0.05:
            # Do not highlight insignificant words
            highlighted_text += f"{word} "
        else:
            highlighted_text += (
                f'<span style="background-color: {color};">{word}</span> '
            )

    # Optional: Replace multiple spaces with a single space and handle newlines
    highlighted_text = re.sub(r"\s+", " ", highlighted_text)
    highlighted_text = highlighted_text.replace("\n", "<br>")

    return highlighted_text.strip()


# --------------------- SHAP Explanation for Both Countries ---------------------
def handle_selection(selected_soc, text_germany, text_switzerland):
    """
    Handles SOC selection from dropdown and returns the SHAP explanations for both Germany and Switzerland.
    """
    if not selected_soc:
        return "Select an SOC from the dropdown to view its SHAP explanations for both countries."

    # Create or retrieve SHAP explainer for the selected SOC
    shap_explainer = create_shap_explainer(selected_soc)
    if shap_explainer is None:
        return "Error creating SHAP explainer for the selected SOC."

    # Generate SHAP explanations
    explanation_germany = explain_soc(shap_explainer, text_germany)
    explanation_switzerland = explain_soc(shap_explainer, text_switzerland)

    # Combine explanations into HTML with separate sections
    combined_explanation = f"""
    <h3>SHAP Explanation for Germany</h3>
    <p>{explanation_germany}</p>
    <h3>SHAP Explanation for Switzerland</h3>
    <p>{explanation_switzerland}</p>
    """

    return combined_explanation


# --------------------- Radar Chart Generation ---------------------
def plot_radar_chart_with_selection(text_germany, text_switzerland):
    """
    Generates the radar chart based on input texts.
    Returns the Plotly figure.
    """
    if not text_germany.strip() or not text_switzerland.strip():
        logger.warning("One or both side effect texts are empty.")
        return go.Figure()  # Return an empty figure or handle accordingly

    scores_germany = classify_adverse_events(text_germany, socs)
    scores_switzerland = classify_adverse_events(text_switzerland, socs)
    fig = plot_radar_chart_plotly(socs, scores_germany, scores_switzerland)
    return fig


# --------------------- Search and Display Functions ---------------------
def search_and_display(drug_name):
    """
    Searches for products in both countries and updates the dropdowns and side effects.

    Args:
        drug_name (str): The name of the drug to search for.

    Returns:
        Tuple: Updates for dropdowns, side effects, links, and comparison section visibility.
    """
    global german_products_dict
    global swiss_products_dict

    # Search in Germany
    try:
        german_products = german_extractor.search_drug(drug_name)
        german_product_names = [product["name"] for product in german_products]
        german_products_dict = {
            product["name"]: product["link"] for product in german_products
        }
        logger.debug(
            f"Found {len(german_product_names)} products in Germany for drug '{drug_name}'."
        )
    except Exception as e:
        logger.error(f"Error fetching German products: {e}")
        german_product_names = []
        german_products_dict = {}

    # Search in Switzerland
    try:
        swiss_products = swiss_extractor.search_drug(drug_name)
        swiss_product_names = [product["name"] for product in swiss_products]
        swiss_products_dict = {
            product["name"]: product["link"] for product in swiss_products
        }
        logger.debug(
            f"Found {len(swiss_product_names)} products in Switzerland for drug '{drug_name}'."
        )
    except Exception as e:
        logger.error(f"Error fetching Swiss products: {e}")
        swiss_product_names = []
        swiss_products_dict = {}

    # Prepare German section updates
    if german_product_names:
        first_german_product = german_product_names[0]
        german_dropdown_update = gr.update(
            choices=german_product_names, value=first_german_product, visible=True
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
        german_dropdown_update = gr.update(choices=[], value=None, visible=True)
        german_side_effects_output_update = gr.update(
            value="No products found in Germany.", visible=True
        )
        german_link_update = gr.update(value="", visible=True)

    # Prepare Swiss section updates
    if swiss_product_names:
        first_swiss_product = swiss_product_names[0]
        swiss_dropdown_update = gr.update(
            choices=swiss_product_names, value=first_swiss_product, visible=True
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
        swiss_dropdown_update = gr.update(choices=[], value=None, visible=True)
        swiss_side_effects_output_update = gr.update(
            value="No products found in Switzerland.", visible=True
        )
        swiss_link_update = gr.update(value="", visible=True)

    # Determine if comparison_section should be visible
    comparison_section_visible = bool(german_product_names) and bool(
        swiss_product_names
    )

    comparison_section_update = gr.update(visible=comparison_section_visible)

    return (
        german_dropdown_update,
        german_side_effects_output_update,
        german_link_update,
        swiss_dropdown_update,
        swiss_side_effects_output_update,
        swiss_link_update,
        comparison_section_update,
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
        try:
            side_effects = german_extractor.get_undesired_effects(product_url)
            # Clean the text if necessary
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
            logger.debug(
                f"Retrieved German side effects for '{selected_product_name}'."
            )
            return side_effects
        except Exception as e:
            logger.error(f"Error retrieving German side effects: {e}")
            return "Unable to retrieve side effects."
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
        try:
            side_effects = swiss_extractor.get_side_effects(product_url)
            # Clean the text if necessary
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
            logger.debug(f"Retrieved Swiss side effects for '{selected_product_name}'.")
            return side_effects
        except Exception as e:
            logger.error(f"Error retrieving Swiss side effects: {e}")
            return "Unable to retrieve side effects."
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
        return f"<a href='{product_url}' target='_blank' style='color:white;'>{product_url}</a>"
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
        return f"<a href='{product_url}' target='_blank' style='color:white;'>{product_url}</a>"
    return ""


# --------------------- Gradio Interface ---------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>DrugSafetyCompare</h1>"
    )

    with gr.Row():
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        search_button = gr.Button("Search")

    # Results Sections
    results_section = gr.Group(visible=True)
    with results_section:
        with gr.Row():
            # German Results Section
            german_section = gr.Column(scale=1, visible=True)
            with german_section:
                gr.HTML("<h3 style='text-align: left;'>Results for Germany</h3>")
                german_dropdown = gr.Dropdown(
                    label="Select a Product (Germany)", choices=[]
                )
                german_side_effects_output = gr.Textbox(
                    label="Undesired Effects (Germany)", lines=10
                )
                german_link_output = gr.HTML()

            # Swiss Results Section
            swiss_section = gr.Column(scale=1, visible=True)
            with swiss_section:
                gr.HTML("<h3 style='text-align: left;'>Results for Switzerland</h3>")
                swiss_dropdown = gr.Dropdown(
                    label="Select a Product (Switzerland)", choices=[]
                )
                swiss_side_effects_output = gr.Textbox(
                    label="Undesired Effects (Switzerland)", lines=10
                )
                swiss_link_output = gr.HTML()

    # Comparison Section
    comparison_section = gr.Group(visible=False)
    with comparison_section:
        gr.HTML("<h2 style='text-align: center;'>Drug Comparison</h2>")

        with gr.Row():
            with gr.Column(scale=1):
                analyze_button = gr.Button("Compare Toxicity Profiles")
                selected_soc = gr.Dropdown(
                    label="Select SOC for SHAP Explanation",
                    choices=[""] + socs,
                    value="",
                    interactive=True,
                )
            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Toxicity Radar Chart")

        with gr.Row():
            explanation_output = gr.HTML(
                label="SHAP Explanation for Selected SOC",
                value="Select an SOC from the dropdown to view its SHAP explanations for both countries.",
                elem_id="explanation-output",
            )

    # Search button logic
    search_button.click(
        fn=search_and_display,
        inputs=drug_input,
        outputs=[
            german_dropdown,
            german_side_effects_output,
            german_link_output,
            swiss_dropdown,
            swiss_side_effects_output,
            swiss_link_output,
            comparison_section,
        ],
    )

    # Compare button logic
    analyze_button.click(
        fn=plot_radar_chart_with_selection,
        inputs=[german_side_effects_output, swiss_side_effects_output],
        outputs=plot_output,
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

    # Define the action when a SOC is selected from the dropdown
    selected_soc.change(
        fn=handle_selection,
        inputs=[selected_soc, german_side_effects_output, swiss_side_effects_output],
        outputs=explanation_output,
    )

    gr.HTML("""
    <div style="text-align: left; font-size: 1em; margin-top: 1em;">
        <p><b>Instructions:</b></p>
        <ol>
            <li>Enter the name of the drug in the search field.</li>
            <li>Click <b>Search</b> to retrieve products from Germany and Switzerland.</li>
            <li>Select a product from each country's dropdown to view its details and link.</li>
            <li>After selecting products, click <b>Compare Toxicity Profiles</b> to generate the radar chart.</li>
            <li>Select an SOC from the dropdown to view its SHAP explanations for both countries.</li>
        </ol>
    </div>
    """)

    # Optional: Add CSS to enhance visibility of the explanation area
    gr.HTML("""
    <style>
    #explanation-output {
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """)


# --------------------- Close Extractors on App Termination ---------------------
def on_close():
    logger.debug("Shutting down extractors.")
    try:
        german_extractor.quit()
    except Exception as e:
        logger.error(f"Error shutting down German extractor: {e}")
    try:
        swiss_extractor.quit()
    except Exception as e:
        logger.error(f"Error shutting down Swiss extractor: {e}")


atexit.register(on_close)


# --------------------- Launch Gradio App ---------------------
def main():
    demo.launch()  # Add share=True if you want to create a public shareable link.

    # http://127.0.0.1:7860/?__theme=light


# Launch the app
if __name__ == "__main__":
    main()
