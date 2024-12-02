"""
Gradio App for DrugSafetyCompare

This script initializes and launches the Gradio web application for drug information retrieval and comparison.

Main functionalities:
- Allows users to search for drugs and retrieve product details from Germany and Switzerland.
- Enables users to compare side effects using two analysis pipelines:
  1. Transformer Pipeline: Uses zero-shot classification and SHAP explainability.
  2. GPT-4 Pipeline: Extracts adverse events using GPT-4 and visualizes them.
- Provides radar charts (for overall toxicity comparison) and sunburst charts (for explainability).

This script only contains UI elements, user interactions, and wiring logic.
All non-UI logic (data retrieval, classification, visualization) is handled in separate modules within `src/`.

Modules used:
- `vianu.drugsafetycompare.src.transformer_pipeline` → Transformer-based classification and SHAP explainability.
- `vianu.drugsafetycompare.src.gpt_pipeline` → GPT-based adverse event extraction and visualization.
- `vianu.drugsafetycompare.src.extract_germany` → German drug information retrieval.
- `vianu.drugsafetycompare.src.extract_switzerland` → Swiss drug information retrieval.
"""

# --------------------- Import Statements ---------------------
import logging
import atexit
import re
import asyncio
import gradio as gr
import torch
import os

# If you have local imports
from dotenv import load_dotenv

# Import your two new pipeline classes
from vianu.drugsafetycompare.src.transformer_pipeline import TransformerPipeline
from vianu.drugsafetycompare.src.gpt_pipeline import GptPipeline

# Import your drug extractors (assumed already existing in ./src)
from vianu.drugsafetycompare.src.extract_germany import GermanDrugInfoExtractor
from vianu.drugsafetycompare.src.extract_switzerland import SwissDrugInfoExtractor


# --------------------- Configure Logging ---------------------
logger = logging.getLogger("drugsafetycompare_logger")

# --------------------- Load Environment Variables ---------------------
load_dotenv()

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

# --------------------- SOC List (could also be in a constants.py) ---------------------
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

# --------------------- Initialize Extractors ---------------------
german_extractor = GermanDrugInfoExtractor()
swiss_extractor = SwissDrugInfoExtractor()

# --------------------- Initialize Pipelines ---------------------
transformer_pipeline = TransformerPipeline(device, socs)
# The GPT pipeline also needs an API key
openai_api_key = os.getenv("OPENAI_TOKEN", "")
gpt_pipeline = GptPipeline(openai_api_key, device, socs)

# --------------------- Global Product Dictionaries ---------------------
german_products_dict = {}
swiss_products_dict = {}

# Store GPT classification results globally (if you prefer)
scores_germany_global = {}
scores_switzerland_global = {}
unique_in_germany_global = {}
unique_in_switzerland_global = {}
color_map_global = {}


# --------------------- Common Functions: Searching/Side Effects Retrieval ---------------------
def get_german_side_effects(selected_product_name):
    if selected_product_name in german_products_dict:
        product_url = german_products_dict[selected_product_name]
        try:
            side_effects = german_extractor.get_undesired_effects(product_url)
            side_effects = re.sub(r"\s+", " ", side_effects).strip()
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


def search_and_display(drug_name):
    global german_products_dict, swiss_products_dict
    error_messages = []

    # Search in Germany
    try:
        german_products = german_extractor.search_drug(drug_name)
        german_product_names = [p["name"] for p in german_products]
        german_products_dict = {p["name"]: p["link"] for p in german_products}
    except Exception as e:
        logger.error(f"Error fetching German products: {e}")
        german_product_names = []
        german_products_dict = {}
        error_messages.append(f"Error fetching German products: {e}")

    # Search in Switzerland
    try:
        swiss_products = swiss_extractor.search_drug(drug_name)
        swiss_product_names = [p["name"] for p in swiss_products]
        swiss_products_dict = {p["name"]: p["link"] for p in swiss_products}
    except Exception as e:
        logger.error(f"Error fetching Swiss products: {e}")
        swiss_product_names = []
        swiss_products_dict = {}
        error_messages.append(f"Error fetching Swiss products: {e}")

    # Prepare updates for the UI elements
    if german_product_names:
        first_german_product = german_product_names[0]
        german_side_effects = get_german_side_effects(first_german_product)
        german_link = update_german_link(first_german_product)

        german_dropdown_update = gr.update(
            choices=german_product_names, value=first_german_product, visible=True
        )
        german_side_effects_output_update = gr.update(
            value=german_side_effects, visible=True
        )
        german_link_update = gr.update(value=german_link, visible=True)
    else:
        german_dropdown_update = gr.update(choices=[], value=None, visible=True)
        german_side_effects_output_update = gr.update(
            value="No products found in Germany.", visible=True
        )
        german_link_update = gr.update(value="", visible=True)

    if swiss_product_names:
        first_swiss_product = swiss_product_names[0]
        swiss_side_effects = get_swiss_side_effects(first_swiss_product)
        swiss_link = update_swiss_link(first_swiss_product)

        swiss_dropdown_update = gr.update(
            choices=swiss_product_names, value=first_swiss_product, visible=True
        )
        swiss_side_effects_output_update = gr.update(
            value=swiss_side_effects, visible=True
        )
        swiss_link_update = gr.update(value=swiss_link, visible=True)
    else:
        swiss_dropdown_update = gr.update(choices=[], value=None, visible=True)
        swiss_side_effects_output_update = gr.update(
            value="No products found in Switzerland.", visible=True
        )
        swiss_link_update = gr.update(value="", visible=True)

    # Show comparison section only if we have at least one product from each
    comparison_section_update = gr.update(
        visible=bool(german_product_names) and bool(swiss_product_names)
    )

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


# --------------------- Gradio App ---------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 style='text-align: center;'>DrugSafetyCompare</h1>")

    # Initial Search Bar
    with gr.Row():
        drug_input = gr.Textbox(label="Enter Drug Name", placeholder="e.g., aspirin")
        search_button = gr.Button("Search")

    error_output = gr.Markdown(value="", visible=False, label="Error Messages")

    results_section = gr.Group(visible=False)
    with results_section:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h3>Results for Germany</h3>")
                german_dropdown = gr.Dropdown(
                    label="Select a Product (Germany)", choices=[]
                )
                german_side_effects_output = gr.Textbox(
                    label="Undesired Effects (Germany)", lines=8, interactive=False
                )
                german_link_output = gr.HTML()

            with gr.Column():
                gr.HTML("<h3>Results for Switzerland</h3>")
                swiss_dropdown = gr.Dropdown(
                    label="Select a Product (Switzerland)", choices=[]
                )
                swiss_side_effects_output = gr.Textbox(
                    label="Undesired Effects (Switzerland)", lines=8, interactive=False
                )
                swiss_link_output = gr.HTML()

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
                value=openai_api_key,
                visible=False,
            )

        analyze_button = gr.Button("Compare Toxicity Profiles")

        # Transformer outputs
        transformer_outputs = gr.Group(visible=True)
        with transformer_outputs:
            plot_output_transformer = gr.Plot()
            selected_soc_transformer = gr.Dropdown(
                label="Select SOC for SHAP Explanation",
                choices=[""] + socs,
                value="",
            )
            explanation_output_transformer = gr.HTML(
                label="SHAP Explanation (Selected SOC)",
                value="Select an SOC from the dropdown to view SHAP explanations.",
            )

        # GPT outputs
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
    <div style="font-size:0.9em;">
      <p><b>Instructions:</b></p>
      <ol>
        <li>Enter the drug name and click <b>Search</b>.</li>
        <li>Select a product from each dropdown (Germany &amp; Switzerland).</li>
        <li>Choose the analysis pipeline (Transformer or GPT-4).</li>
        <li>Click <b>Compare Toxicity Profiles</b> to generate comparisons.</li>
      </ol>
    </div>
    """)

    # Wiring up the search button
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

    # Update side effects based on selected product
    german_dropdown.change(
        fn=lambda p: (get_german_side_effects(p), update_german_link(p)),
        inputs=german_dropdown,
        outputs=[german_side_effects_output, german_link_output],
    )
    swiss_dropdown.change(
        fn=lambda p: (get_swiss_side_effects(p), update_swiss_link(p)),
        inputs=swiss_dropdown,
        outputs=[swiss_side_effects_output, swiss_link_output],
    )

    # Switch pipeline (shows/hides GPT vs. Transformer UI elements)
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

    # Perform comparison
    def analyze_pipeline(pipeline_choice, text_germany, text_switzerland, user_api_key):
        if pipeline_choice == "Transformer Pipeline":
            # Transformer pipeline logic
            scores_germany = transformer_pipeline.classify_adverse_events(text_germany)
            scores_switzerland = transformer_pipeline.classify_adverse_events(
                text_switzerland
            )
            fig_transformer = transformer_pipeline.plot_radar_chart(
                socs, scores_germany, scores_switzerland
            )

            # Hide GPT results, show only the transformer chart
            return (
                gr.update(visible=True, value=fig_transformer),
                gr.update(visible=False),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        else:
            # GPT pipeline logic
            async def async_analyze_gpt():
                # Update local pipeline's api_key if needed
                gpt_pipeline.api_key = user_api_key

                # Extract AE from Germany
                (
                    adverse_events_germany,
                    error_germany,
                ) = await gpt_pipeline.get_ae_from_openai_async(text_germany)
                (
                    adverse_events_switzerland,
                    error_switzerland,
                ) = await gpt_pipeline.get_ae_from_openai_async(text_switzerland)

                if error_germany or error_switzerland:
                    return (
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(
                            value=f"{error_germany}\n{error_switzerland}", visible=True
                        ),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )

                # Classify
                gpt_pipeline.scores_germany = gpt_pipeline.classify_adverse_events(
                    adverse_events_germany
                )
                gpt_pipeline.scores_switzerland = gpt_pipeline.classify_adverse_events(
                    adverse_events_switzerland
                )

                # Generate color map & unique sets
                color_map = {
                    **gpt_pipeline.generate_color_map(gpt_pipeline.scores_germany),
                    **gpt_pipeline.generate_color_map(gpt_pipeline.scores_switzerland),
                }
                unique_in_germany, unique_in_switzerland = (
                    gpt_pipeline.identify_unique_adverse_events(
                        gpt_pipeline.scores_germany, gpt_pipeline.scores_switzerland
                    )
                )

                # Make radar chart
                fig_radar = gpt_pipeline.plot_radar_chart(
                    socs,
                    {
                        s: d["normalized_score"]
                        for s, d in gpt_pipeline.scores_germany.items()
                    },
                    {
                        s: d["normalized_score"]
                        for s, d in gpt_pipeline.scores_switzerland.items()
                    },
                )
                # Make sunburst
                fig_sunburst = gpt_pipeline.draw_sunburst_with_highlights(
                    gpt_pipeline.scores_germany,
                    gpt_pipeline.scores_switzerland,
                    unique_in_germany,
                    unique_in_switzerland,
                    selected_soc=None,
                    highlighted_only=False,
                    color_map=color_map,
                )

                # Store globally (if needed later)
                global scores_germany_global, scores_switzerland_global
                global unique_in_germany_global, unique_in_switzerland_global
                global color_map_global
                scores_germany_global = gpt_pipeline.scores_germany
                scores_switzerland_global = gpt_pipeline.scores_switzerland
                unique_in_germany_global = unique_in_germany
                unique_in_switzerland_global = unique_in_switzerland
                color_map_global = color_map

                return (
                    gr.update(visible=False),
                    gr.update(visible=True, value=fig_radar),
                    gr.update(value="", visible=False),
                    gr.update(visible=True, value=fig_sunburst),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            return asyncio.run(async_analyze_gpt())

    analyze_button.click(
        fn=analyze_pipeline,
        inputs=[
            pipeline_selector,
            german_side_effects_output,
            swiss_side_effects_output,
            api_key_input,
        ],
        outputs=[
            # Transformer outputs
            plot_output_transformer,
            # GPT outputs
            plot_output_radar_gpt,
            error_output,
            plot_output_sunburst_gpt,
            selected_soc_gpt,
            highlight_toggle_gpt,
        ],
    )

    # SHAP explanation logic for Transformer
    def handle_shap_explanation(selected_soc, text_germany, text_switzerland):
        return transformer_pipeline.explain_soc(
            selected_soc, text_germany, text_switzerland
        )

    selected_soc_transformer.change(
        fn=handle_shap_explanation,
        inputs=[
            selected_soc_transformer,
            german_side_effects_output,
            swiss_side_effects_output,
        ],
        outputs=explanation_output_transformer,
    )

    # Update sunburst chart for GPT pipeline
    def update_sunburst_wrapper(selected_soc, show_highlighted_only):
        return gpt_pipeline.update_sunburst_chart(
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
