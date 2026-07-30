# Ads Marketing Campaign

An end-to-end machine learning pipeline for advertising campaign analysis and revenue prediction.

This repository includes data preprocessing, feature engineering, feature selection, and model training modules, along with an Apache Airflow DAG for orchestration and exploratory notebooks for analysis.

## Key Features

- Preprocess raw ad campaign data and split into training and test sets
- Create derived features such as aggregated clicks and leads
- Select target and feature columns based on configuration
- Train a Decision Tree regression model with MLflow tracking and local artifact saving
- Orchestrate pipeline steps using an Airflow DAG
- Explore data and feature engineering in Jupyter notebooks

## Repository Structure

- `config/`
  - `01_config.yaml`: example pipeline configuration with dataset paths, output paths, and feature selection settings
  - `02_config.yaml`: additional configuration example
  - `pipeline_config.yaml`: schema-style configuration for storage and feature selection
- `dags/`
  - `01_pipeline_dag.py`: Airflow DAG definition for pipeline orchestration
- `notebooks/`
  - `01_exploratory_data_analysis.ipynb`
  - `02_feature_engineering.ipynb`
- `src/`
  - `pipeline_initialization/initialize.py`: pipeline initialization and output directory creation
  - `data_preprocessing/preprocess.py`: raw data preprocessing and train/test split
  - `feature_engineering/create_features.py`: feature creation step
  - `feature_engineering/select_features.py`: feature selection step
  - `modelling/training/train_decision_tree.py`: Decision Tree model training with MLflow tracking
  - `modelling/ensembling/ensemble.py`: ensemble model helpers and ensemble strategies
  - `modelling/evaluation/evaluate.py`: model evaluation utilities
  - `utils/logger.py`: logging utility for pipeline execution
- `requirements.txt`: pinned Python dependencies used by the project
- `pyproject.toml`: package metadata and install configuration

## Prerequisites

- Python 3.11+ (Apache Airflow 3.x requires Python 3.11 or later)
- Git
- Recommended: virtual environment for Python package isolation

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install -e .
```

## Configuration

Before running the pipeline, update `config/01_config.yaml` to match your local repository paths and dataset location.

The example configuration currently expects:

- raw data file: `data/raw/ads_campaign_data.csv`
- processed data: written under the configured `output/` directory
- selected feature files and model artifacts written under configured subdirectories

If you use a different working directory or dataset location, adjust the YAML paths accordingly.

## Running the Pipeline (Individual run)

### Initialize the pipeline

```bash
python3 src/pipeline_initialization/initialize.py -c config/01_config.yaml
```

### Preprocess raw data

```bash
python3 src/data_preprocessing/preprocess.py -c config/01_config.yaml
```

### Create new features

```bash
python3 src/feature_engineering/create_features.py -c config/01_config.yaml
```

### Select features for modeling

```bash
python3 src/feature_engineering/select_features.py -c config/01_config.yaml
```

### Train the model

```bash
python3 src/modelling/training/train_decision_tree.py -c config/01_config.yaml
```

## Apache Airflow Orchestration

The `dags/01_pipeline_dag.py` file defines a simple Airflow DAG for the initialization, preprocessing, feature creation, and feature selection steps.

To use the DAG:

1. Install and initialize Airflow if needed.
2. Place the DAG file in your Airflow DAGs folder.
3. Update `CONFIG_FILE_PATH` in `dags/01_pipeline_dag.py` to point to your configuration file.
4. Start the Airflow scheduler and webserver.

## Notebooks

Use the notebooks to inspect data and feature engineering decisions:

- `notebooks/01_exploratory_data_analysis.ipynb`
- `notebooks/02_feature_engineering.ipynb`

## Notes

- The project does not include the raw dataset. Add your own dataset file in `data/raw/` or update the config paths accordingly.
- Output directories and file paths are controlled by the YAML configuration.
- The current feature engineering pipeline creates a `Clicks_Plus_Leads` feature and selects features based on the configuration. More to follow
- The model training script logs metrics and artifacts via MLflow and saves a local model backup.
