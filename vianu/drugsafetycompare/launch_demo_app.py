"""
Merged Gradio App for Drug Information Scraping and Comparison.

This application allows users to:
1. Search for a drug and retrieve its products from Germany and Switzerland.
2. Select specific products to view their side effects.
3. Choose between two analysis pipelines:
   - Transformer Pipeline: Uses zero-shot classification and SHAP explanations.
   - GPT-4 Pipeline: Extracts adverse events using GPT-4, classifies SOCs, and visualizes with sunburst charts.
4. Compare the side effects using radar charts and respective explainability methods.
"""

# --------------------- Import Statements ---------------------
import logging
import gradio as gr
import torch
from transformers import pipeline
import plotly.graph_objects as go
import numpy as np
import re
import sys
import os
import asyncio
import shap
import html
import atexit
import ast
import plotly.subplots as sp
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI

from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor
from vianu.drugsafetycompare.src.switzerland import SwissDrugInfoExtractor

# Adjust the path according to your project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


# --------------------- Load Environment Variables ---------------------
load_dotenv()

# --------------------- Configure Logging ---------------------
logger = logging.getLogger("drug_compare_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------- Initialize Extractors ---------------------
german_extractor = GermanDrugInfoExtractor()
swiss_extractor = SwissDrugInfoExtractor()

# --------------------- Global Product Dictionaries ---------------------
german_products_dict = {}
swiss_products_dict = {}

# --------------------- Determine Device ---------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA device")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("Using MPS device")
else:
    device = torch.device("cpu")
    logger.info("Using CPU device")

# --------------------- Initialize Classifier ---------------------
try:
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device.type != "cpu" else -1,
    )
    logger.info("Zero-shot classifier initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing zero-shot classifier: {e}")
    classifier = None

# --------------------- Define SOCs ---------------------
socs = [
    "Blood and lymphatic system disorders",
    "Cardiac disorders",
    "Congenital, familial and genetic disorders",
    "Ear and labyrinth disorders",
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
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "Nervous system disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Product issues",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Reproductive system and breast disorders",
    "Respiratory, thoracic and mediastinal disorders",
    "Skin and subcutaneous tissue disorders",
    "Social circumstances",
    "Surgical and medical procedures",
    "Vascular disorders",
]


# --------------------- Caching Decorators ---------------------
@lru_cache(maxsize=128)
def cache_search_drug_extractor(extractor_class, drug_name):
    extractor = extractor_class()
    products = extractor.search_drug(drug_name)
    return products


# --------------------- Transformer Pipeline Functions ---------------------


def classify_adverse_events_transformer(text, candidate_labels):
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


def plot_radar_chart_transformer(socs, scores_a, scores_b):
    """
    Plots a radar chart using Plotly comparing two sets of scores across multiple categories.
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

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(scores_a.values()), max(scores_b.values()), 0.6)],
            ),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                showline=True,
                showticklabels=True,
                tickfont=dict(size=10),
                linewidth=1,
                tickangle=0,
            ),
        ),
        margin=dict(l=100, r=100, t=100, b=100),
        showlegend=True,
        title="Comparison of Drug Toxicity Profiles by System Organ Class (SOC)",
        height=700,
        width=700,
    )

    return fig


def soc_model(texts, soc):
    """
    Model function for SHAP that returns scores for a specific SOC.
    """
    logger.debug(f"soc_model called with texts: {texts}, soc: {soc}")
    if isinstance(texts, str):
        texts = [texts]
    scores = []
    for txt in texts:
        try:
            result = classifier(txt, [soc], multi_label=True)
            if isinstance(result, dict):
                score = result["scores"][0] if result["scores"] else 0.0
            elif isinstance(result, list) and len(result) > 0:
                score = result[0]["scores"][0] if result[0]["scores"] else 0.0
            else:
                score = 0.0
            scores.append(score)
        except Exception as e:
            logger.error(f"Error during classification in soc_model: {e}")
            scores.append(0.0)
    return np.array(scores)


# Initialize SHAP Explainer per SOC and cache them
shap_explainers = {}


def create_shap_explainer(soc):
    """
    Creates or retrieves a SHAP explainer for the specified SOC.
    """
    if soc not in shap_explainers:
        try:
            # Define a function instead of using a lambda to fix 'soc' parameter
            def model_fn(texts):
                return soc_model(texts, soc)

            shap_explainer = shap.Explainer(model_fn, masker=shap.maskers.Text())
            shap_explainers[soc] = shap_explainer
            logger.debug(f"SHAP explainer created for SOC: {soc}")
        except Exception as e:
            logger.error(f"Error creating SHAP explainer for SOC '{soc}': {e}")
            return None
    return shap_explainers[soc]


# --------------------- explain_soc Function ---------------------
def explain_soc(shap_explainer, text_germany, text_switzerland):
    """
    Generates SHAP explanations for the given text using the provided SHAP explainer.
    Returns the explanation as an HTML formatted string with highlighted words.
    """
    # Check if text is empty
    if not text_germany.strip() and not text_switzerland.strip():
        return "No side effect descriptions available for analysis."

    explanations = []
    for country, text in [("Germany", text_germany), ("Switzerland", text_switzerland)]:
        if not text.strip():
            explanation = f"No side effect descriptions available for {country}."
        else:
            try:
                shap_values = shap_explainer([text])[0]
            except Exception as e:
                logger.error(f"Error generating SHAP values: {e}")
                explanation = f"Error generating SHAP explanations for {country}."
                explanations.append(f"<h3>{country}</h3><p>{explanation}</p>")
                continue

            words = shap_values.data
            shap_values_list = shap_values.values

            if not words.size or not shap_values_list.size:
                explanation = (
                    f"No significant words found for SHAP explanations for {country}."
                )
                explanations.append(f"<h3>{country}</h3><p>{explanation}</p>")
                continue

            # Build the HTML string with highlighted words
            max_val = np.max(np.abs(shap_values_list))
            min_val = np.min(np.abs(shap_values_list))
            # To avoid division by zero
            range_val = max_val - min_val if max_val - min_val != 0 else 1
            explanation = ""
            for word, value in zip(words, shap_values_list):
                # Normalize value between 0 and 1
                normalized_value = (abs(value) - min_val) / range_val
                # Map the SHAP value to a color
                if value > 0:
                    color = f"rgba(255, 0, 0, {normalized_value})"  # Red for positive SHAP values
                else:
                    color = f"rgba(0, 0, 255, {normalized_value})"  # Blue for negative SHAP values
                explanation += f"<span style='background-color: {color}'>{html.escape(word)}</span> "
            explanation = f"<p>{explanation}</p>"

        explanations.append(f"<h3>{country}</h3>{explanation}")

    combined_explanation = "<br>".join(explanations)
    return combined_explanation


def handle_shap_explanation(selected_soc, text_germany, text_switzerland):
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
    explanation = explain_soc(shap_explainer, text_germany, text_switzerland)

    return explanation


# --------------------- GPT-4 Pipeline Functions ---------------------


async def get_ae_from_openai_async(text, api_key):
    """
    Asynchronously fetch adverse events from OpenAI.
    """
    prompt = """
You are an expert assistant trained to extract specific information from text. Given the following text, return a Python list of all adverse events and side effects mentioned in the text. Provide only the Python list as your output, without any additional explanations or text.

Example 1:
Input:
"The most commonly reported side effects include headache, nausea, and fatigue. Rare side effects include hair loss and blurred vision."

Output:
["Headache", "Nausea", "Fatigue", "Hair loss", "Blurred vision"]

Example 2:
Input:
"Patients have reported experiencing skin rash, dry mouth, and difficulty breathing after taking this medication. In rare cases, seizures have also been observed."

Output:
["Skin rash", "Dry mouth", "Difficulty breathing", "Seizures"]

Example 3:
Input:
"This drug may cause dizziness, stomach upset, and in some instances, temporary memory loss. It has also been linked to muscle pain and joint stiffness."

Output:
["Dizziness", "Stomach upset", "Temporary memory loss", "Muscle pain", "Joint stiffness"]

Now, analyze the following text and return a Python list of all adverse events and side effects:
"""

    try:
        client = OpenAI(api_key=api_key)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        content_list = ast.literal_eval(content)  # Safely parse the list
        logger.info(f"Got {len(content_list)} AE from OpenAI.")
        return content_list, ""
    except Exception as e:
        error_msg = f"Failed to extract adverse events: {e}"
        logger.error(error_msg)
        return [], error_msg


def classify_adverse_events_gpt(adverse_events, candidate_labels):
    """
    Classify adverse events and explain predictions by showing contributing AEs, cumulative scores, and normalized scores.

    Args:
        adverse_events (List[str]): List of extracted adverse events.
        candidate_labels (List[str]): List of SOCs to classify into.

    Returns:
        Dict[str, Dict]: Detailed classification results for each SOC, including:
            - Contributing adverse events (list of strings).
            - Cumulative score.
            - Normalized score.
    """
    if not adverse_events:
        return {
            soc: {
                "adverse_events": [],
                "cumulative_score": 0.0,
                "normalized_score": 0.0,
            }
            for soc in candidate_labels
        }

    if classifier is None:
        logger.error("Zero-shot classifier is not initialized.")
        return {
            soc: {
                "adverse_events": [],
                "cumulative_score": 0.0,
                "normalized_score": 0.0,
            }
            for soc in candidate_labels
        }

    soc_data = {
        soc: {"adverse_events": [], "cumulative_score": 0.0} for soc in candidate_labels
    }

    try:
        for event in adverse_events:
            result = classifier(event, candidate_labels, multi_label=True)
            max_label = result["labels"][np.argmax(result["scores"])]
            for label, score in zip(result["labels"], result["scores"]):
                if label == max_label:
                    soc_data[label]["adverse_events"].append(event)
                soc_data[label]["cumulative_score"] += score

        # Normalize scores to range [0, 1]
        max_score = max(soc["cumulative_score"] for soc in soc_data.values()) or 1
        for soc in soc_data:
            soc_data[soc]["normalized_score"] = (
                soc_data[soc]["cumulative_score"] / max_score
            )

        logger.debug(f"Detailed SOC classification data: {soc_data}")
        return soc_data
    except Exception as e:
        logger.error(f"Error during SOC classification: {e}")
        return {
            soc: {
                "adverse_events": [],
                "cumulative_score": 0.0,
                "normalized_score": 0.0,
            }
            for soc in candidate_labels
        }


def plot_radar_chart_gpt(socs, scores_a, scores_b):
    """
    Plots a radar chart using Plotly comparing two sets of scores across multiple categories.
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

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],  # Since scores are normalized
            ),
        ),
        showlegend=True,
        title="Comparison of Drug Toxicity Profiles by SOC",
    )

    return fig


def generate_color_map(scores):
    color_map = {}
    soc_list = list(scores.keys())
    colors = [
        "#AEC6CF",
        "#FFB347",
        "#B39EB5",
        "#617bff",
        "#77DD77",
        "#FDFD96",
        "#CFCFC4",
        "#FFCCCB",
        "#F49AC2",
        "#BC8F8F",
        "#F5DEB3",
        "#D8BFD8",
        "#E6E6FA",
        "#FFDAB9",
        "#F0E68C",
        "#DAE8FC",
        "#ACE1AF",
        "#FFE4E1",
        "#ADD8E6",
        "#D4AF37",
        "#FFC0CB",
        "#D9F3FF",
        "#FFEBCD",
        "#E3A857",
        "#BAED91",
        "#D6D6D6",
        "#FFEFD5",
        "#DEB887",
        "#FFD1DC",
        "#C8A2C8",
    ]
    for idx, soc in enumerate(soc_list):
        base_color = colors[idx % len(colors)]
        color_map[soc] = base_color
        for event in scores[soc]["adverse_events"]:
            color_map[event] = base_color
    return color_map


def identify_unique_adverse_events(scores_germany, scores_switzerland):
    unique_in_germany = {}
    unique_in_switzerland = {}

    all_socs = set(scores_germany.keys()).union(scores_switzerland.keys())
    for soc in all_socs:
        events_germany = set(scores_germany.get(soc, {}).get("adverse_events", []))
        events_switzerland = set(
            scores_switzerland.get(soc, {}).get("adverse_events", [])
        )

        unique_in_germany[soc] = events_germany - events_switzerland
        unique_in_switzerland[soc] = events_switzerland - events_germany

    return unique_in_germany, unique_in_switzerland


def draw_sunburst_with_highlights(
    scores_germany,
    scores_switzerland,
    unique_in_germany,
    unique_in_switzerland,
    selected_soc=None,
    highlighted_only=False,
    color_map={},
):
    # Prepare data for Germany
    labels_germany = []
    parents_germany = []
    values_germany = []
    marker_colors_germany = []

    for soc, data in scores_germany.items():
        if selected_soc and soc != selected_soc:
            continue
        soc_events = (
            unique_in_germany.get(soc, [])
            if highlighted_only
            else data["adverse_events"]
        )
        if not soc_events:
            continue
        labels_germany.append(soc)
        parents_germany.append("")
        values_germany.append(len(soc_events))
        marker_colors_germany.append(color_map.get(soc, "#FFFFFF"))

        for event in soc_events:
            labels_germany.append(event)
            parents_germany.append(soc)
            values_germany.append(1)
            if event in unique_in_germany.get(soc, []):
                marker_colors_germany.append("red")  # Unique events in Germany
            else:
                marker_colors_germany.append(color_map.get(event, "#FFFFFF"))

    # Prepare data for Switzerland
    labels_switzerland = []
    parents_switzerland = []
    values_switzerland = []
    marker_colors_switzerland = []

    for soc, data in scores_switzerland.items():
        if selected_soc and soc != selected_soc:
            continue
        soc_events = (
            unique_in_switzerland.get(soc, [])
            if highlighted_only
            else data["adverse_events"]
        )
        if not soc_events:
            continue
        labels_switzerland.append(soc)
        parents_switzerland.append("")
        values_switzerland.append(len(soc_events))
        marker_colors_switzerland.append(color_map.get(soc, "#FFFFFF"))

        for event in soc_events:
            labels_switzerland.append(event)
            parents_switzerland.append(soc)
            values_switzerland.append(1)
            if event in unique_in_switzerland.get(soc, []):
                marker_colors_switzerland.append("red")  # Unique events in Switzerland
            else:
                marker_colors_switzerland.append(color_map.get(event, "#FFFFFF"))

    # Create the plot
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=["Germany", "Switzerland"],
    )

    if labels_germany and parents_germany and values_germany:
        fig.add_trace(
            go.Sunburst(
                labels=labels_germany,
                parents=parents_germany,
                values=values_germany,
                branchvalues="total",
                hoverinfo="label+value",
                marker=dict(colors=marker_colors_germany),
            ),
            row=1,
            col=1,
        )

    if labels_switzerland and parents_switzerland and values_switzerland:
        fig.add_trace(
            go.Sunburst(
                labels=labels_switzerland,
                parents=parents_switzerland,
                values=values_switzerland,
                branchvalues="total",
                hoverinfo="label+value",
                marker=dict(colors=marker_colors_switzerland),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
    return fig


def update_sunburst(
    selected_soc,
    show_highlighted_only,
    scores_germany,
    scores_switzerland,
    unique_in_germany,
    unique_in_switzerland,
    color_map,
):
    """
    Updates the sunburst charts based on selected SOC and highlight toggle.

    Args:
        selected_soc (str): Selected SOC or "All".
        show_highlighted_only (bool): Toggle to show only highlighted events.
        scores_germany (dict): SOC classification scores for Germany.
        scores_switzerland (dict): SOC classification scores for Switzerland.
        unique_in_germany (dict): Unique adverse events in Germany.
        unique_in_switzerland (dict): Unique adverse events in Switzerland.
        color_map (dict): Mapping of SOCs and events to colors.

    Returns:
        plotly.graph_objects.Figure: Updated sunburst chart.
    """
    if selected_soc == "All":
        selected_soc = None

    fig = draw_sunburst_with_highlights(
        scores_germany,
        scores_switzerland,
        unique_in_germany,
        unique_in_switzerland,
        selected_soc,
        show_highlighted_only,
        color_map=color_map,
    )
    return fig


# --------------------- Common Functions ---------------------


def get_german_side_effects(selected_product_name):
    """
    Retrieves side effects for the selected German product.
    """
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        try:
            side_effects = german_extractor.get_undesired_effects(product_url)
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
            logger.info(f"Retrieved German side effects for '{selected_product_name}'.")
            return side_effects
        except Exception as e:
            logger.error(f"Error retrieving German side effects: {e}")
            return "Unable to retrieve side effects."
    else:
        return "Product not found."


def get_swiss_side_effects(selected_product_name):
    """
    Retrieves side effects for the selected Swiss product.
    """
    if selected_product_name in swiss_products_dict:
        product_url = swiss_products_dict[selected_product_name]
        try:
            side_effects = swiss_extractor.get_side_effects(product_url)
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
            logger.info(f"Retrieved Swiss side effects for '{selected_product_name}'.")
            return side_effects
        except Exception as e:
            logger.error(f"Error retrieving Swiss side effects: {e}")
            return "Unable to retrieve side effects."
    else:
        return "Product not found."


def update_german_link(selected_product_name):
    """
    Updates the German product link.
    """
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""


def update_swiss_link(selected_product_name):
    """
    Updates the Swiss product link.
    """
    if selected_product_name in swiss_products_dict:
        product_url = swiss_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""


def search_and_display(drug_name):
    """
    Searches for the drug in both German and Swiss databases.
    Returns updates for the Gradio interface components.
    """
    global german_products_dict
    global swiss_products_dict

    # Initialize error messages
    error_messages = []

    # Search in Germany
    try:
        german_products = german_extractor.search_drug(drug_name)
        german_product_names = [product["name"] for product in german_products]
        german_products_dict = {
            product["name"]: product["link"] for product in german_products
        }
        logger.info(
            f"Found {len(german_product_names)} products in Germany for drug '{drug_name}'."
        )
    except Exception as e:
        logger.error(f"Error fetching German products: {e}")
        german_product_names = []
        german_products_dict = {}
        error_messages.append(f"Error fetching German products: {e}")

    # Search in Switzerland
    try:
        swiss_products = swiss_extractor.search_drug(drug_name)
        swiss_product_names = [product["name"] for product in swiss_products]
        swiss_products_dict = {
            product["name"]: product["link"] for product in swiss_products
        }
        logger.info(
            f"Found {len(swiss_product_names)} products in Switzerland for drug '{drug_name}'."
        )
    except Exception as e:
        logger.error(f"Error fetching Swiss products: {e}")
        swiss_product_names = []
        swiss_products_dict = {}
        error_messages.append(f"Error fetching Swiss products: {e}")

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

    # Update error messages if any
    if error_messages:
        return (
            german_dropdown_update,
            german_side_effects_output_update,
            german_link_update,
            swiss_dropdown_update,
            swiss_side_effects_output_update,
            swiss_link_update,
            comparison_section_update,
            gr.update(value="\n".join(error_messages), visible=True),
        )
    else:
        return (
            german_dropdown_update,
            german_side_effects_output_update,
            german_link_update,
            swiss_dropdown_update,
            swiss_side_effects_output_update,
            swiss_link_update,
            comparison_section_update,
            gr.update(value="", visible=False),
        )


# --------------------- Gradio Interface ---------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>DrugSafetyCompare</h1>"
    )

    # Initial Search Bar
    with gr.Row():
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        search_button = gr.Button("Search")

    # Error Message Display
    error_output = gr.Markdown(
        value="",
        visible=False,
        label="Error Messages",
        elem_id="error-output",
        show_label=False,
    )

    # Results Sections (Initially Hidden)
    results_section = gr.Group(visible=False)
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
                    label="Undesired Effects (Germany)", lines=10, interactive=False
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
                    label="Undesired Effects (Switzerland)", lines=10, interactive=False
                )
                swiss_link_output = gr.HTML()

    # Comparison Section (Initially Hidden)
    comparison_section = gr.Group(visible=False)
    with comparison_section:
        gr.HTML("<h2 style='text-align: center;'>Drug Comparison</h2>")
        with gr.Row():
            pipeline_selector = gr.Radio(
                choices=["Transformer Pipeline", "GPT-4 Pipeline"],
                label="Select Analysis Pipeline",
                value="Transformer Pipeline",
            )
            api_key_input = gr.Textbox(
                label="OpenAI API Key (Required for GPT-4 Pipeline)",
                type="password",
                placeholder="Enter your OpenAI API key here",
                value=os.getenv("OPENAI_TOKEN", ""),
                visible=False,
            )

        with gr.Row():
            analyze_button = gr.Button("Compare Toxicity Profiles")

        # Transformer Pipeline Outputs
        transformer_outputs = gr.Group(visible=True)
        with transformer_outputs:
            plot_output_transformer = gr.Plot()
            selected_soc_transformer = gr.Dropdown(
                label="Select SOC for SHAP Explanation",
                choices=[""] + socs,
                value="",
                interactive=True,
            )
            explanation_output_transformer = gr.HTML(
                label="SHAP Explanation for Selected SOC",
                value="Select an SOC from the dropdown to view its SHAP explanations for both countries.",
                elem_id="explanation-output",
            )

        # GPT-4 Pipeline Outputs
        gpt_outputs = gr.Group(visible=False)
        with gpt_outputs:
            plot_output_radar_gpt = gr.Plot()
            with gr.Row():
                selected_soc_gpt = gr.Dropdown(
                    label="Select SOC", choices=["All"] + socs, value="All"
                )
                highlight_toggle_gpt = gr.Checkbox(
                    label="Show Only Differences", value=False
                )
            plot_output_sunburst_gpt = gr.Plot()

    gr.HTML("""
    <div style="text-align: left; font-size: 1em; margin-top: 1em;">
        <p><b>Instructions:</b></p>
        <ol>
            <li>Enter the name of the drug in the search field.</li>
            <li>Click <b>Search</b> to retrieve products from Germany and Switzerland.</li>
            <li>Select a product from each country's dropdown to view its details and link.</li>
            <li>Select an analysis pipeline (Transformer or GPT-4).</li>
            <li>Click <b>Compare Toxicity Profiles</b> to generate the charts.</li>
            <li>Use the provided controls to explore the explanations.</li>
        </ol>
    </div>
    """)

    # Search button logic
    def make_results_visible():
        return gr.update(visible=True)

    search_button.click(
        fn=make_results_visible,
        inputs=None,
        outputs=results_section,
    ).then(
        fn=search_and_display,
        inputs=[drug_input],
        outputs=[
            german_dropdown,
            german_side_effects_output,
            german_link_output,
            swiss_dropdown,
            swiss_side_effects_output,
            swiss_link_output,
            comparison_section,
            error_output,
        ],
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

    # Pipeline selector logic
    def update_pipeline(selected_pipeline):
        if selected_pipeline == "GPT-4 Pipeline":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

    pipeline_selector.change(
        fn=update_pipeline,
        inputs=pipeline_selector,
        outputs=[api_key_input, transformer_outputs, gpt_outputs],
    )

    # Analyze button logic
    def analyze_pipeline(pipeline_choice, text_germany, text_switzerland, api_key):
        if pipeline_choice == "Transformer Pipeline":
            # Transformer pipeline logic
            scores_germany = classify_adverse_events_transformer(text_germany, socs)
            scores_switzerland = classify_adverse_events_transformer(
                text_switzerland, socs
            )
            fig_transformer = plot_radar_chart_transformer(
                socs, scores_germany, scores_switzerland
            )
            # Hide GPT outputs
            return (
                gr.update(visible=True, value=fig_transformer),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        else:
            # GPT-4 pipeline logic
            # Since this involves async functions, we need to use asyncio.run
            async def async_analyze():
                adverse_events_germany, error_germany = await get_ae_from_openai_async(
                    text_germany, api_key
                )
                (
                    adverse_events_switzerland,
                    error_switzerland,
                ) = await get_ae_from_openai_async(text_switzerland, api_key)
                if error_germany or error_switzerland:
                    error_msg = f"{error_germany}\n{error_switzerland}"
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=error_msg, visible=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )
                scores_germany = classify_adverse_events_gpt(
                    adverse_events_germany, socs
                )
                scores_switzerland = classify_adverse_events_gpt(
                    adverse_events_switzerland, socs
                )
                global scores_germany_global, scores_switzerland_global
                global unique_in_germany_global, unique_in_switzerland_global
                global color_map_global
                scores_germany_global = scores_germany
                scores_switzerland_global = scores_switzerland
                color_map_global = {
                    **generate_color_map(scores_germany),
                    **generate_color_map(scores_switzerland),
                }
                unique_in_germany_global, unique_in_switzerland_global = (
                    identify_unique_adverse_events(scores_germany, scores_switzerland)
                )
                fig_radar = plot_radar_chart_gpt(
                    socs,
                    {
                        soc: data["normalized_score"]
                        for soc, data in scores_germany.items()
                    },
                    {
                        soc: data["normalized_score"]
                        for soc, data in scores_switzerland.items()
                    },
                )
                fig_sunburst = draw_sunburst_with_highlights(
                    scores_germany,
                    scores_switzerland,
                    unique_in_germany_global,
                    unique_in_switzerland_global,
                    selected_soc=None,
                    highlighted_only=False,
                    color_map=color_map_global,
                )
                # Hide Transformer outputs
                return (
                    gr.update(visible=False),
                    gr.update(visible=True, value=fig_radar),
                    gr.update(visible=False),
                    gr.update(visible=True, value=fig_sunburst),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            return asyncio.run(async_analyze())

    analyze_button.click(
        fn=analyze_pipeline,
        inputs=[
            pipeline_selector,
            german_side_effects_output,
            swiss_side_effects_output,
            api_key_input,
        ],
        outputs=[
            plot_output_transformer,
            plot_output_radar_gpt,
            error_output,
            plot_output_sunburst_gpt,
            selected_soc_gpt,
            highlight_toggle_gpt,
        ],
    )

    # SHAP explanation logic for transformer pipeline
    selected_soc_transformer.change(
        fn=handle_shap_explanation,
        inputs=[
            selected_soc_transformer,
            german_side_effects_output,
            swiss_side_effects_output,
        ],
        outputs=explanation_output_transformer,
    )

    # Update sunburst chart based on selections for GPT-4 pipeline
    def update_sunburst_wrapper(selected_soc, show_highlighted_only):
        return update_sunburst(
            selected_soc,
            show_highlighted_only,
            scores_germany_global,
            scores_switzerland_global,
            unique_in_germany_global,
            unique_in_switzerland_global,
            color_map_global,
        )

    highlight_toggle_gpt.change(
        fn=update_sunburst_wrapper,
        inputs=[selected_soc_gpt, highlight_toggle_gpt],
        outputs=plot_output_sunburst_gpt,
    )

    selected_soc_gpt.change(
        fn=update_sunburst_wrapper,
        inputs=[selected_soc_gpt, highlight_toggle_gpt],
        outputs=plot_output_sunburst_gpt,
    )


# --------------------- Close Extractors on App Termination ---------------------
def on_close():
    logger.info("Shutting down extractors.")
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
    demo.launch()


if __name__ == "__main__":
    main()
