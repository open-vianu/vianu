# Spotting Clinical Knowledge (SpoCK)

## Setup
```bash
git clone https://github.com/open-vianu/vianu.git
cd vianu
poetry install
poetry shell
```

### Requirements specific for SpoCK
- `aiohttp`
- `dacite`
- `gradio`
- `numpy`


## Demos
Run a demo pipeline
```bash
python vianu/spock/launch_demo_pipeline.py
```

Run a demo app 
```bash
python vianu/spock/launch_demo_app.py
```

## CLI
```bash
alias spock="python -m vianu.spock"
```

```bash
spock --source pubmed --source ema --term dafalgan --log-level DEBUG --n-ner-tasks 1
```
