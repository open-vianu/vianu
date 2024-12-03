import gradio as gr

# Processing search input
def process_search(search_text):
    response = f"Case '{search_text}' processing"
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
    color: #6366f1; 
    background: #E0E7FF;
    border-radius: 10px;
    font: Montserrat;
    font-size: 50px;
    font-weight: bold;
    text-align: center;
}
"""

# Layout resembling the image
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image(value="vianu/spock/assets/spock_logo.png", show_label=False, elem_id="logo-image")
        with gr.Column(scale=5):
            gr.Markdown("<div id='title-text'>SpoCK: Spott Clinical Knowledge</div>")
                
    with gr.Row():
        search_input = gr.Textbox(label="Search", placeholder="Enter your search here...")
            
    with gr.Row():
        search_results = gr.Textbox(lines=10, label="Recently searched", interactive=False)
            
    search_input.submit(
        fn=process_search,
        inputs=search_input,
        outputs=search_results
    )   

if __name__ == '__main__':
    demo.launch()
