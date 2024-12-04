from argparse import Namespace

import gradio as gr

from vianu.spock.src import scraping as scp
from vianu.spock.src import chunking as cnk
from vianu.spock.src import ner


namespace_kwargs = {
    "min_chunk_size": 500,
    "min_chunk_overlap": 50,
    "source": "pubmed",
    "model": "ollama",
}


# Processing search input
def process_pipeline(search_text):
    args = Namespace(term=search_text, **namespace_kwargs)
    data = []
    scp.apply(args_=args, data=data, save_data=False)
    cnk.apply(args_=args, data=data, save_data=False)
    data = data[:1]
    ner.apply(args_=args, data=data, save_data=False)
    text_entity = data[0].text_entities[0]
    med_prod = ' '.join([ne.text for ne in text_entity.medicinal_products])
    adv_react = ' '.join([ne.text for ne in text_entity.adverse_reactions])
    response = f"{text_entity.text}\n\nMedicinal products: {med_prod}\n\nAdverse reactions: {adv_react}"
    return response

custom_css = """ 
#logo-image {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    height: 20vh;
}

#title-text {
    display: flex;
    justify-content: center;
    align-items: center; 
    height: 100%; 
    color: var(--block-title-text-color);
    background: var(--block-title-background-fill);
    border-radius: 10px;
    font: Montserrat;
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}
"""

# Layout resembling the image
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    scraping_state = gr.State(value=False)
    ner_state = gr.State(value=False)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="vianu/spock/assets/spock_logo_circular.png", show_label=False, elem_id="logo-image")
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spotting Clinical Knowledge</div>")
                
    with gr.Row():
        search_input = gr.Textbox(label="Search", placeholder="Enter your search here...")
            
    with gr.Row():
        search_results = gr.Textbox(lines=10, label="Recently searched", interactive=False)
            
    search_input.submit(
        fn=process_pipeline,
        inputs=search_input,
        outputs=search_results
    )   

if __name__ == '__main__':
    demo.launch()
