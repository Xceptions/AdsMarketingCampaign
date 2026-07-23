import os
import sys
from pathlib import Path
from datetime import datetime
from airflow.sdk import DAG, task


ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_preprocessing import preprocess
from src.feature_engineering import create_features, select_features

CONFIG_FILE_PATH = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/config/pipeline_config.yaml'

with DAG(
    dag_id="id_pipeline_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["machine_learning"],
) as dag:

    @task
    def preprocess_data_task(config_file: str):
        """Executes data preprocessing script using the pipeline configuration"""
        preprocess.main(config_path=config_file)
        return config_file

    @task
    def create_features_task(config_file: str):
        """Executes feature creation script using the pipeline configuration"""
        create_features.main(config_path=config_file)
        return config_file

    @task
    def select_features_task(config_file: str):
        """Executes feature selection script using the pipeline configuration"""
        select_features.main(config_path=config_file)
        return config_file


    config_pointer = preprocess_data_task(CONFIG_FILE_PATH)
    config_pointer = create_features_task(config_pointer)
    config_pointer = select_features_task(config_pointer)