from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import List

import gradio as gr


HEAD_FILE = Path(__file__).parent / "assets/head/scripts.html"
CSS_FILE = Path(__file__).parent / "assets/css/styles.css"
SPOCK_KWARGS = {
    "term": "dafalgan", 
    "source": ["pubmed"], 
    "model": "llama", 
    "ner_tasks": 2,
    "log_level": "DEBUG", 
}




# Layout resembling the image
with gr.Blocks(
    head_paths=HEAD_FILE, css_paths=CSS_FILE, theme=gr.themes.Soft()
) as demo:
    scraping_state = gr.State(value=False)
    ner_state = gr.State(value=False)

    with gr.Row(elem_id="logo-title-row"):
        with gr.Column(scale=1):
            gr.Image(
                value="vianu/spock/assets/images/spock_logo_circular.png",
                show_label=False,
                elem_id="logo-image",
            )
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spotting Clinical Knowledge</div>")

    with gr.Row():
        search_input = gr.Textbox(
            label="Search", placeholder="Enter your search here..."
        )

    with gr.Row():
        search_results = gr.HTML(label="Recently searched").change()

    with gr.Row():
        gr.HTML('<div id="details" class="details-container"></div>')

    search_input.submit(
        fn=_process_pipeline, inputs=search_input, outputs=search_results
    )

if __name__ == "__main__":
    demo.launch()
