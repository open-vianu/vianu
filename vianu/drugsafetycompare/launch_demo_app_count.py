"""
Merged Gradio App for Drug Information Scraping and Comparison with Adverse Event Extraction.

This application allows users to:
1. Search for a drug and retrieve its products from Germany and Switzerland.
2. Select specific products to view their side effects.
3. Extract adverse events from the side effects using OpenAI's GPT-4.
4. Compare the side effects using SOC classification and visualize them with radar charts and sunburst plots.
"""

import logging
import gradio as gr
import torch
from transformers import pipeline
import plotly.graph_objects as go
import numpy as np
import re
import sys
import os
import plotly.subplots as sp
from openai import OpenAI
import ast
import atexit
from dotenv import load_dotenv

load_dotenv()

# --------------------- Configure Logging ---------------------
logger = logging.getLogger("drug_compare_logger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --------------------- Import Extractors ---------------------
from vianu.drugsafetycompare.src.germany import GermanDrugInfoExtractor
from vianu.drugsafetycompare.src.switzerland import SwissDrugInfoExtractor

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
        "zero-shot-classification", model="facebook/bart-large-mnli", device=device
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

# --------------------- Define Functions ---------------------

def get_ae_from_openai(text, api_key):
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
        client = OpenAI(
            api_key=api_key
        )
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
        )
        # Accessing the message content correctly
        content = response.choices[0].message.content
        logger.info(content)
        content_list = ast.literal_eval(content)  # Safely convert the string representation of a Python list to an actual list
        return content_list, ""
    except Exception as e:
        error_msg = f"Error in get_ae_from_openai: {e}"
        logger.error(error_msg)
        return [], error_msg

def classify_adverse_events(adverse_events, candidate_labels):
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

def plot_radar_chart_plotly(socs, scores_a, scores_b):
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
    soc_list = scores.keys()
    colors = [
        "#AEC6CF", "#FFB347", "#B39EB5", "#617bff", "#77DD77",
        "#FDFD96", "#CFCFC4", "#FFCCCB", "#F49AC2", "#BC8F8F",
        "#F5DEB3", "#D8BFD8", "#E6E6FA", "#FFDAB9", "#F0E68C",
        "#DAE8FC", "#ACE1AF", "#FFE4E1", "#ADD8E6", "#D4AF37",
        "#FFC0CB", "#D9F3FF", "#FFEBCD", "#E3A857", "#BAED91",
        "#D6D6D6", "#FFEFD5", "#DEB887", "#FFD1DC", "#C8A2C8",
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

def update_charts_highlighted(selected_soc, show_highlighted_only, scores_germany, scores_switzerland, unique_in_germany, unique_in_switzerland, color_map):
    if selected_soc == "All":
        selected_soc = None

    fig = draw_sunburst_with_highlights(
        scores_germany,
        scores_switzerland,
        unique_in_germany,
        unique_in_switzerland,
        selected_soc,
        show_highlighted_only,
        color_map
    )
    return fig

def draw_sunburst_with_highlights(
    scores_germany,
    scores_switzerland,
    unique_in_germany,
    unique_in_switzerland,
    selected_soc=None,
    highlighted_only=False,
    color_map={}
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

def plot_radar_chart_with_selection(text_germany, text_switzerland, api_key):
    """
    Generates the radar chart and sunburst chart based on input texts.
    Returns the Plotly figures and any error messages.
    """
    global scores_germany_global, scores_switzerland_global
    global unique_in_germany_global, unique_in_switzerland_global
    global color_map

    error_messages = []

    if not text_germany.strip() or not text_switzerland.strip():
        logger.warning("One or both side effect texts are empty.")
        error_messages.append("One or both side effect texts are empty.")
        return go.Figure(), go.Figure(), gr.update(value="\n".join(error_messages), visible=True)

    # Extract adverse events
    adverse_events_germany, error_germany = get_ae_from_openai(text_germany, api_key)
    adverse_events_switzerland, error_switzerland = get_ae_from_openai(text_switzerland, api_key)

    if error_germany:
        error_messages.append(f"Germany Side Effects Extraction Error: {error_germany}")
    if error_switzerland:
        error_messages.append(f"Switzerland Side Effects Extraction Error: {error_switzerland}")

    # If there are errors, return them without proceeding further
    if error_messages:
        return go.Figure(), go.Figure(), gr.update(value="\n".join(error_messages), visible=True)

    # Classify SOCs based on adverse events
    scores_germany = classify_adverse_events(adverse_events_germany, socs)
    scores_switzerland = classify_adverse_events(adverse_events_switzerland, socs)

    # Store globally for use in other functions
    scores_germany_global = scores_germany
    scores_switzerland_global = scores_switzerland

    # Generate color map
    color_map = {
        **generate_color_map(scores_germany),
        **generate_color_map(scores_switzerland),
    }

    # Identify unique adverse events
    unique_in_germany, unique_in_switzerland = identify_unique_adverse_events(
        scores_germany, scores_switzerland
    )

    # Store globally
    unique_in_germany_global = unique_in_germany
    unique_in_switzerland_global = unique_in_switzerland

    # Radar chart
    fig_radar = plot_radar_chart_plotly(
        socs,
        {soc: data["normalized_score"] for soc, data in scores_germany.items()},
        {soc: data["normalized_score"] for soc, data in scores_switzerland.items()},
    )

    # Sunburst chart
    initial_fig_sunburst = draw_sunburst_with_highlights(
        scores_germany,
        scores_switzerland,
        unique_in_germany,
        unique_in_switzerland,
        selected_soc=None,
        highlighted_only=False,
        color_map=color_map
    )

    return fig_radar, initial_fig_sunburst, gr.update(value="", visible=False)

def search_and_display(drug_name):
    global german_products_dict
    global swiss_products_dict

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
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        try:
            side_effects = german_extractor.get_undesired_effects(product_url)
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
            logger.info(
                f"Retrieved German side effects for '{selected_product_name}'."
            )
            return side_effects
        except Exception as e:
            logger.error(f"Error retrieving German side effects: {e}")
            return "Unable to retrieve side effects."
    else:
        return "Product not found."

def get_swiss_side_effects(selected_product_name):
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
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""

def update_swiss_link(selected_product_name):
    if selected_product_name in swiss_products_dict:
        product_url = swiss_products_dict[selected_product_name]
        return f"<a href='{product_url}' target='_blank'>{product_url}</a>"
    return ""

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

# --------------------- Gradio Interface ---------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<h1 style='text-align: center; font-size: 2em; font-weight: bold;'>DrugSafetyCompare</h1>"
    )

    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            type="password",
            placeholder="Enter your OpenAI API key here",
            value=os.getenv("OPENAI_TOKEN") or "",
            interactive=True,
        )
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        search_button = gr.Button("Search")

    # Error Message Display
    error_output = gr.Markdown(
        value="", 
        visible=False, 
        label="Error Messages",
        elem_id="error-output"
    )

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
            selected_soc = gr.Dropdown(
                label="Select SOC", choices=["All"] + socs, value="All"
            )
            highlight_toggle = gr.Checkbox(label="Show Only Highlighted", value=False)
        plot_output_radar = gr.Plot()
        plot_output_sunburst = gr.Plot()

    # Search button logic
    search_button.click(
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
        ],
    ).then(
        fn=plot_radar_chart_with_selection,
        inputs=[german_side_effects_output, swiss_side_effects_output, api_key_input],
        outputs=[plot_output_radar, plot_output_sunburst, error_output],
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

    # Update sunburst chart based on selections
    def update_sunburst(selected_soc, show_highlighted_only):
        fig = update_charts_highlighted(
            selected_soc,
            show_highlighted_only,
            scores_germany_global,
            scores_switzerland_global,
            unique_in_germany_global,
            unique_in_switzerland_global,
            color_map
        )
        return fig

    highlight_toggle.change(
        fn=update_sunburst,
        inputs=[selected_soc, highlight_toggle],
        outputs=plot_output_sunburst,
    )

    selected_soc.change(
        fn=update_sunburst,
        inputs=[selected_soc, highlight_toggle],
        outputs=plot_output_sunburst,
    )

    gr.HTML("""
    <div style="text-align: left; font-size: 1em; margin-top: 1em;">
        <p><b>Instructions:</b></p>
        <ol>
            <li>Enter your OpenAI API key in the designated field.</li>
            <li>Enter the name of the drug in the search field.</li>
            <li>Click <b>Search</b> to retrieve products from Germany and Switzerland.</li>
            <li>Select a product from each country's dropdown to view its details and link.</li>
        </ol>
    </div>
    """)

# --------------------- Main Function ---------------------
def main():
    demo.launch(inbrowser=True)

if __name__ == "__main__":
    main()