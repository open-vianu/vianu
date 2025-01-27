"""
GPT-4 Pipeline for Adverse Event Extraction and Classification

This module implements the GPT-4-powered analysis pipeline used in the DrugSafetyCompare app.

Main functionalities:
- Calls OpenAI’s GPT-4 API to extract adverse events from free-text drug descriptions.
- Classifies the extracted adverse events into System Organ Classes (SOCs) using a zero-shot classifier.
- Normalizes and visualizes classification results via radar charts (Plotly).
- Uses sunburst charts to display adverse events grouped by SOC, highlighting differences between Germany and Switzerland.
- Identifies unique adverse events for each country and color-codes them.

This module does NOT handle UI elements; it is used by `launch_demo_app.py` for AI-powered drug safety comparisons.

Dependencies:
- `openai`
- `transformers`
- `plotly`
- `numpy`
- `asyncio`
"""

import logging
import ast
import asyncio
import numpy as np

from openai import OpenAI
from transformers import pipeline

logger = logging.getLogger("drugsafetycompare_logger")


class GptPipeline:
    """
    Handles all logic for the GPT-based pipeline, including calls to the OpenAI API,
    classification with zero-shot, and the radar/sunburst chart creation.
    """

    def __init__(self, api_key, device, socs):
        self.api_key = api_key
        self.device = device
        self.socs = socs

        # Initialize zero-shot classifier
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device.type != "cpu" else -1,
            )
            logger.info("Zero-shot classifier (for GPT pipeline) initialized.")
        except Exception as e:
            logger.error(
                f"Error initializing zero-shot classifier for GPT pipeline: {e}"
            )
            self.classifier = None

        # We keep these dicts at instance level so we can dynamically update them
        self.scores_germany = {}
        self.scores_switzerland = {}
        self.unique_in_germany = {}
        self.unique_in_switzerland = {}
        self.color_map = {}

    async def get_ae_from_openai_async(self, text):
        """
        Asynchronously fetch adverse events from OpenAI using GPT.
        Returns (list_of_AEs, error_message).
        """
        prompt = """
You are an expert assistant trained to extract specific information from text. 
Given the following text, return a Python list of all adverse events and side effects 
mentioned in the text. Provide only the Python list as output, no extra text.

Example Input:
"The most commonly reported side effects include headache, nausea, and fatigue."

Expected Output:
["Headache", "Nausea", "Fatigue"]
        """.strip()

        if not text.strip():
            return ([], "")

        try:
            client = OpenAI(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",  # Adjust if needed
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            content_list = ast.literal_eval(content)
            logger.info(f"Retrieved {len(content_list)} AEs from GPT.")
            return (content_list, "")
        except Exception as e:
            error_msg = f"Failed to extract adverse events: {e}"
            logger.error(error_msg)
            return ([], error_msg)

    def classify_adverse_events(self, adverse_events):
        """
        Classify adverse events into SOCs using zero-shot classification.
        Returns a dictionary with structure:
            {
              SOC: {
                "adverse_events": [...],
                "cumulative_score": float,
                "normalized_score": float
              }, ...
            }
        """
        if not adverse_events:
            return {
                soc: {
                    "adverse_events": [],
                    "cumulative_score": 0.0,
                    "normalized_score": 0.0,
                }
                for soc in self.socs
            }

        if self.classifier is None:
            logger.error("Zero-shot classifier is not initialized for GPT pipeline.")
            return {
                soc: {
                    "adverse_events": [],
                    "cumulative_score": 0.0,
                    "normalized_score": 0.0,
                }
                for soc in self.socs
            }

        soc_data = {
            soc: {"adverse_events": [], "cumulative_score": 0.0} for soc in self.socs
        }

        for event in adverse_events:
            try:
                result = self.classifier(event, self.socs, multi_label=True)
                max_label = result["labels"][np.argmax(result["scores"])]
                for label, score in zip(result["labels"], result["scores"]):
                    # add event to whichever label is the top label
                    if label == max_label:
                        soc_data[label]["adverse_events"].append(event)
                    soc_data[label]["cumulative_score"] += score
            except Exception as e:
                logger.error(f"Error classifying event '{event}': {e}")

        # Normalize
        max_score = max(s["cumulative_score"] for s in soc_data.values()) or 1.0
        for soc in soc_data:
            soc_data[soc]["normalized_score"] = (
                soc_data[soc]["cumulative_score"] / max_score
            )

        return soc_data

    def plot_radar_chart(self, socs, scores_a, scores_b):
        """
        Create a radar chart for GPT pipeline results (scores are normalized).
        """
        import plotly.graph_objects as go

        categories = socs + [socs[0]]
        values_a = [scores_a.get(soc, 0) for soc in socs] + [scores_a.get(socs[0], 0)]
        values_b = [scores_b.get(soc, 0) for soc in socs] + [scores_b.get(socs[0], 0)]

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
                radialaxis=dict(visible=True, range=[0, 1]),
            ),
            showlegend=True,
            title="GPT Pipeline: Comparison of Drug Toxicity Profiles by SOC",
        )
        return fig

    def generate_color_map(self, scores):
        """
        Generate a color map for SOCs and events for the sunburst chart.
        """
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
        color_map = {}
        soc_list = list(scores.keys())
        for idx, soc in enumerate(soc_list):
            base_color = colors[idx % len(colors)]
            color_map[soc] = base_color
            for event in scores[soc]["adverse_events"]:
                color_map[event] = base_color
        return color_map

    def identify_unique_adverse_events(self, scores_germany, scores_switzerland):
        """
        Identify events unique to Germany or Switzerland.
        """
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
        self,
        scores_germany,
        scores_switzerland,
        unique_in_germany,
        unique_in_switzerland,
        selected_soc=None,
        highlighted_only=False,
        color_map={},
    ):
        import plotly.subplots as sp
        import plotly.graph_objects as go

        # Prepare data for Germany
        labels_germany = []
        parents_germany = []
        values_germany = []
        marker_colors_germany = []

        # Build Germany data
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

        # Build Swiss data
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
                    marker_colors_switzerland.append(
                        "red"
                    )  # Unique events in Switzerland
                else:
                    marker_colors_switzerland.append(color_map.get(event, "#FFFFFF"))

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

    def update_sunburst_chart(
        self,
        selected_soc,
        show_highlighted_only,
        scores_germany,
        scores_switzerland,
        unique_in_germany,
        unique_in_switzerland,
        color_map,
    ):
        """
        Returns an updated sunburst figure based on the selected SOC and highlight toggle.
        """
        if selected_soc == "All":
            selected_soc = None

        return self.draw_sunburst_with_highlights(
            scores_germany,
            scores_switzerland,
            unique_in_germany,
            unique_in_switzerland,
            selected_soc=selected_soc,
            highlighted_only=show_highlighted_only,
            color_map=color_map,
        )
