---
title: SpoCK
layout: home
---

# Welcome to **SpoCK** 🚀
**SpoCK** is a user-friendly [Gradio](https://www.gradio.app/) application designed to search public websites for 

`Sp`otting `C`linical `K`nowledge. 

It brings together two main steps to streamline your exploration:  

- **`scraping`**: Extract relevant text from public websites  
- **`ner`**: Use LLMs to identify *drug names* and *adverse drug reactions*

### Why SpoCK?  
Built with an `async` concurrency framework, SpoCK ensures efficient performance and simplifies horizontal scaling; this aims to address the handling of growing data and user demands.

**SpoCK** is aimed to be highliy customizable. Here are some ideas adapt it to your needs:  

- 🔍 **Add a database**: Store processed information for easy reference and analysis.  
- 👥 **Enable login/user management**: Personalize user experiences.  
- 🧠 **Expand entity recognition**: Search for more or different named entities tailored to your needs.  
- 🌐 **Broaden your search**: Integrate additional or alternative data sources for even richer insights.  

Give **SpoCK** a try and see how it transforms clinical knowledge discovery into a breeze! 🌟  


### Getting Started
#### Prerequisites
- Python 3.11 or higher
- OpenAI API Key or Ollama Enpoint

A local Ollama inference container can be set up by:
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker exec -it ollama ollama pull llama3.2
```

#### Python environment
```bash
git clone https://github.com/open-vianu/vianu.git
cd vianu
poetry install
poetry shell
```

**Requirements specific for SpoCK**
- `aiohttp`
- `beautifulsoup4`
- `dacite`
- `gradio`
- `numpy`
- `pymupdf`
- `python-dotenv`
- `openai`
- `defusedxml`

#### Environment Variables
You can use an `.env` file for defining the following Envornment Variables:
- `OLLAMA_BASE_URL`: when using Ollama Endpoint
- `OPENAI_API_KEY`: when using OpenAI Endpoint

or alternatively set the corresponding environment variable through the UI.

### Run demo pipeline
```bash
python vianu/spock/launch_demo_pipeline.py
```

### Run demo app
```bash
python vianu/spock/launch_demo_app.py
```

### CLI
```bash
python -m vianu.spock --term dafalgan --model llama --data-path "/tmp/spock" --data-file "spock_data" --log-level DEBUG
```

### Disclaimer
This project is intended for educational and personal use only. Users are required to respect the terms and conditions, 
`robots.txt` rules, and any other access policies of the websites they interact with.

- Please use this tool responsibly and ethically.
- Do not send excessive requests that could overwhelm servers or negatively impact the performance of the targeted websites.
- Before scraping, always verify that your activities comply with the website’s policies and local laws.
- The creators of this project assume no liability for the misuse of this tool.

By using this project, you agree to adhere to these guidelines and accept full responsibility for your actions.


