<p align="center">
    <img src="https://avatars.githubusercontent.com/u/189356226?s=400&u=4bc88c9f31bc573f84d4222461c520e19c1c97a4&v=4" alt="vianu Logo" width="150" height="150" />
    <h3 align="center">vianu</h3>
    <p align="center">Together we are building an Open Source Community for Life Sciences.</p>
    <p align="center">
        <a href="https://badge.fury.io/py/vianu"><img alt="PyPI version" src="https://badge.fury.io/py/vianu.svg?icon=si%3Apython"></a>
    </p>
</p>

---

Vianu is a Python package designed for developers working in the **life sciences and healthcare** sectors. It provides access to a variety of tools and workflows, allowing users to quickly build, validate, and deploy data-driven applications.

## Available Tools

- **Lasa**: A tool for phonetic comparison of novel drug names with authorized ones from different locations.
- **FraudCrawler**: A data ingestion and transformation pipeline for real-world healthcare data.
- **SpoCK**: A tool to search public websites for spotting adverse drug reactions
- **DrugSafetyCompare**: A tool to search for published drug labels and compare their safety profiles. [See in action](https://huggingface.co/spaces/vianu/drugsafetycompare)

## Installation

To install Vianu, use the following command:

```bash
pip install vianu
```

Alternatively, you can install vianu from source:

```bash
git clone https://github.com/smc40/vianu.git
cd vianu
poetry install
poetry shell
```


## Usage
### Lasa
TBD
#### Launch a Demo App

#### Launch a Demo Pipeline

```bash
cd vianu/lasa
poetry install
poetry shell
python -m vianu.tools.tici.launch_demo_pipeline
```

### FraudCrawler

#### Launch a Demo App

```bash
python vianu/fraudcrawler/launch_demo_app.py
```

#### Launch a Demo Pipeline

```bash
python -m vianu.fraudcrawler.launch_demo_pipeline
```


### DrugSafetyCompare
**DrugSafetyCompare** is a tool designed to help users search for drugs, retrieve product information from Germany and Switzerland, extract adverse events, and compare side effects using SOC (System Organ Class) classification. Leveraging OpenAI's GPT-4 and interactive visualization tools, DrugSafetyCompare provides a comprehensive overview of drug safety profiles.

#### Launch a Demo App
To lauch the demo app do the following:

```bash
poetry install
poetry shell
python vianu/tools/drugsafetycompare/launch_demo_app_count.py
```
Or simply launch the starter-script:
```bash
vianu_drugsafetycompare_app

```

#### Launch a Demo Pipeline

```bash
poetry install
poetry shell
python -m vianu.drugsafetycompare.launch_demo_pipeline
```


### SpoCK

#### Launch a Demo App

```bash
python vianu/spock/launch_demo_app.py
```

#### Launch a Demo Pipeline

```bash
python -m vianu.spock.launch_demo_pipeline
```


## Run tests
To run all tests with a covarage report run:
```bash
pytest --cov-report=html tests/tests-*/* -s
```




## Contributing

We welcome contributions from the community! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## Code of Conduct

We strive to create a welcoming environment for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for more details.

## License

This project is licensed under the MIT License.

## Feedback and Support

For any issues or feature requests, please use the [GitHub Issues](https://github.com/smc40/vianu/issues) page.

## Acknowledgments

We thank all contributors and the broader open-source community for their support and feedback.