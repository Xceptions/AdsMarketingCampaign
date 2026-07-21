import os
import sys
from pathlib import Path
import yaml
import argparse
import logging
import numpy as np
import pandas as pd

# logs should be saved to db not file
# file used here is placeholder
log_dir = Path("/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/logs")
log_file = log_dir / "production.log"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

class Preprocess:
    """ Preprocessing based on Exploration """
    
    def __init__(self, data_path:str, output_path:str) -> None:
        self.data_path = data_path
        self.output_path = output_path
        self.file_path = Path(data_path)
        self.file_name = self.file_path.stem
        self.file_ext = self.file_path.suffix

    def read_data(self, data_path: str):
        """Read data using pandas or spark"""
        df = pd.read_csv(data_path)
        return df

    def preprocess(self, df: pd.DataFrame) -> None:
        """
        Preprocessing the data using pandas or spark.
        Cleaning, imputation
        """
        # using first 20k as train and next 5k as test
        df_train = df[:20000]
        df_test = df[20000:25000]
        return df_train, df_test

    def save_data(self, 
                df: pd.DataFrame,
                output_path: str,
                name: str,
                version: str,
                ext: str) -> bool:
        return df.to_csv(f'{output_path}{name}_{version}{ext}', index=False)
        
    def run_step(self) -> bool:
        self.df = self.read_data(self.data_path)
        self.df_train, self.df_test = self.preprocess(self.df)
        self.save_data(self.df_train, self.output_path, self.file_name, 'train', self.file_ext)
        self.save_data(self.df_test, self.output_path, self.file_name, 'test', self.file_ext)
        logging.info(f'Process complete. Outputs saved to {self.output_path}')

        return True

def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_path = config['storage']['base_data']
    output_path = config['storage']['processed_dir']

    preprocess = Preprocess(
        data_path = data_path,
        output_path = output_path
    )
    preprocess.run_step()


if __name__ == "__main__":
    # check if file is called using another yaml file
    # python3 preprocess.py --config path_to_file <- absolute is better
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()

    # if path is relative, convert to absolute
    config_path = Path(args.config).resolve()

    logging.info(f"Resolving configuration path: {config_path}")

    if not config_path.exists():
        logging.error(f"Configuration file missing: {config_path}")
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    logging.info(f'Successfully loaded configuration file at {config_path}')

    main(config_path)