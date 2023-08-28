# Machine Learning Project Template

*A template repository to help ramp-up machine learning modeling and experimentation.*

## Setup

### Dependencies

- python >= 3.8
- pip >= 21.0
- docker >= 20.10

### Installation

1. Build module: `pip install .`
1. Install requirements: `pip install -r requirements.txt`

## Usage

### Options for running the ML pipeline

- Using the bash script: `bash scripts/training/pipeline.sh`
- Using the prefect flow: `python scripts/training/pipeline.py`

#### Using the prefect orchestration locally

1. Set prefect to use a local server: `prefect backend server`
1. Start prefect server: `prefect server start`
1. Start a prefect server agent: `prefect agent local start`
1. Register prefect flow: `python scripts/training/pipeline.py --register true`
1. Navigate to the prefect UI to review or schedule the flow: [prefect](http://127.0.0.1:8080)

### Review Run Metrics

1. Launch MLFlow UI: `mlflow ui`
1. Navigate to the UI: [mlflow](http://127.0.0.1:5000)

### Launch Model Server

1. Build and Run Container: `./scripts/serving/scripts/run.sh`

### Development

- Run tests: `nox`

## Technologies

### Machine Learning

- [sklearn](https://scikit-learn.org/0.21/documentation.html) - general machine learning
- [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - neural networks

### Model Server

- [docker](https://docs.docker.com/) - server container
- [fastapi](https://fastapi.tiangolo.com/) - web framework

### MLOps

- [mlflow](https://www.mlflow.org/docs/latest/index.html) - model management
- [prefect](https://docs.prefect.io/) - worfklow management

### Testing

- [nox](https://nox.thea.codes/en/stable/) - test automation
- [flake8](https://flake8.pycqa.org/en/latest/) - linter
- [black](https://black.readthedocs.io/en/stable/) - formatter
- [mypy](https://mypy.readthedocs.io/en/stable/getting_started.html) - static type checker
- [pytest](https://docs.pytest.org/en/stable/contents.html) - testing
