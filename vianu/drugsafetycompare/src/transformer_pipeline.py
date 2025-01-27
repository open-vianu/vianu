"""
Transformer Pipeline for Adverse Event Classification and Explainability

This module implements the Transformer-based analysis pipeline used in the DrugSafetyCompare app.

Main functionalities:
- Uses `facebook/bart-large-mnli` for zero-shot classification of adverse events into System Organ Classes (SOCs).
- Normalizes and visualizes classification results via radar charts (using Plotly).
- Implements SHAP (SHapley Additive Explanations) to provide word-level explainability for each SOC.
- Provides a caching mechanism to store SHAP explanations per SOC.

This module does NOT handle UI elements; it only contains functions for classification and visualization.
It is used by `launch_demo_app.py` for drug safety comparisons.

Dependencies:
- `transformers`
- `torch`
- `shap`
- `plotly`
"""

import logging
import html
import numpy as np
import shap

from transformers import pipeline

logger = logging.getLogger("drugsafetycompare_logger")


class TransformerPipeline:
    """
    Handles all logic for the Transformer-based (zero-shot + SHAP) pipeline.
    """

    def __init__(self, device, socs):
        """
        Initialize the zero-shot classifier and other necessary resources.
        """
        self.device = device
        self.socs = socs
        self.shap_explainers = {}

        # Initialize the zero-shot classifier
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device.type != "cpu" else -1,
            )
            logger.info("Zero-shot classifier initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing zero-shot classifier: {e}")
            self.classifier = None

    def classify_adverse_events(self, text):
        """
        Classify the given text into multiple candidate SOC labels using zero-shot classification.
        Returns a dictionary of SOCs with their corresponding normalized scores.
        """
        if not text.strip():
            return {soc: 0.0 for soc in self.socs}

        if self.classifier is None:
            logger.error("Zero-shot classifier is not initialized.")
            return {soc: 0.0 for soc in self.socs}

        try:
            result = self.classifier(text, self.socs, multi_label=True)
            scores = dict(zip(result["labels"], result["scores"]))
            max_score = max(scores.values()) if scores else 1.0
            normalized_scores = {
                soc: score / max_score for soc, score in scores.items()
            }
            return normalized_scores
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return {soc: 0.0 for soc in self.socs}

    def plot_radar_chart(self, socs, scores_a, scores_b):
        """
        Plots a radar chart (Plotly) comparing two sets of scores across multiple categories (SOCs).
        Returns the Plotly Figure object.
        """
        import plotly.graph_objects as go

        categories = socs.copy()
        categories += [socs[0]]  # close the loop for radar

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
                radialaxis=dict(
                    visible=True,
                    range=[
                        0,
                        max(
                            max(scores_a.values(), default=0),
                            max(scores_b.values(), default=0),
                            0.6,
                        ),
                    ],
                ),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            showlegend=True,
            title="Comparison of Drug Toxicity Profiles by System Organ Class (SOC)",
            height=700,
            width=700,
        )

        return fig

    def _soc_model(self, texts, soc):
        """
        Model function for SHAP that returns scores for a specific SOC.
        """
        if isinstance(texts, str):
            texts = [texts]
        scores = []
        for txt in texts:
            try:
                result = self.classifier(txt, [soc], multi_label=True)
                score = result["scores"][0] if result["scores"] else 0.0
                scores.append(score)
            except Exception as e:
                logger.error(f"Error during classification for SHAP: {e}")
                scores.append(0.0)
        return np.array(scores)

    def _create_shap_explainer(self, soc):
        """
        Creates or retrieves a SHAP explainer for the specified SOC.
        """
        if soc not in self.shap_explainers:
            try:
                # define a function that partially applies 'soc' for the model
                def model_fn(texts):
                    return self._soc_model(texts, soc)

                shap_explainer = shap.Explainer(model_fn, masker=shap.maskers.Text())
                self.shap_explainers[soc] = shap_explainer
            except Exception as e:
                logger.error(f"Error creating SHAP explainer for SOC '{soc}': {e}")
                return None
        return self.shap_explainers[soc]

    def explain_soc(self, selected_soc, text_germany, text_switzerland):
        """
        Generates SHAP explanations for the specified SOC and the two texts.
        Returns an HTML string containing the highlighted text.
        """
        if not selected_soc:
            return "Select an SOC from the dropdown to view SHAP explanations."

        shap_explainer = self._create_shap_explainer(selected_soc)
        if shap_explainer is None:
            return "Error creating SHAP explainer for the selected SOC."

        # If both texts are empty
        if not text_germany.strip() and not text_switzerland.strip():
            return "No side effect descriptions available for analysis."

        explanations = []

        for country, text_data in [
            ("Germany", text_germany),
            ("Switzerland", text_switzerland),
        ]:
            if not text_data.strip():
                explanations.append(
                    f"<h3>{country}</h3><p>No side effect descriptions available.</p>"
                )
                continue

            try:
                shap_values = shap_explainer([text_data])[0]
            except Exception as e:
                logger.error(f"Error generating SHAP values: {e}")
                explanations.append(
                    f"<h3>{country}</h3><p>Error generating SHAP explanations.</p>"
                )
                continue

            words = shap_values.data
            shap_vals = shap_values.values

            if not words.size or not shap_vals.size:
                explanations.append(
                    f"<h3>{country}</h3><p>No significant words found.</p>"
                )
                continue

            max_val = np.max(np.abs(shap_vals))
            min_val = np.min(np.abs(shap_vals))
            range_val = max_val - min_val if (max_val - min_val) != 0 else 1.0

            highlight_html = ""
            for word, value in zip(words, shap_vals):
                normalized_value = (abs(value) - min_val) / range_val
                if value > 0:
                    color = f"rgba(255,0,0,{normalized_value})"
                else:
                    color = f"rgba(0,0,255,{normalized_value})"
                highlight_html += f"<span style='background-color: {color}'>{html.escape(word)}</span> "

            explanations.append(f"<h3>{country}</h3><p>{highlight_html}</p>")

        return "<br>".join(explanations)
