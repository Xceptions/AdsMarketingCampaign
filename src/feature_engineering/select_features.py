import os
import sys
from pathlib import Path
from typing import Tuple
import yaml
import argparse
import logging
import numpy as np
import pandas as pd

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

class SelectFeatures:
    """ Feature selection based on Exploration """
    
    def __init__(self, train_data_path:str, test_data_path:str, output_path:str) -> None:
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_path = output_path
        
        # extract file name and extension for saving later
        self.train_file_path = Path(train_data_path)
        self.train_file_name = self.train_file_path.stem
        self.train_file_ext = self.train_file_path.suffix

        self.test_file_path = Path(test_data_path)
        self.test_file_name = self.test_file_path.stem
        self.test_file_ext = self.test_file_path.suffix

    def read_data(self, train_data_path:str, test_data_path:str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Read data using pandas or spark """
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        return df_train, df_test

    def select_features(self, df_train:pd.DataFrame, df_test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Selecting features for training """
        features = ['Clicks_Plus_Leads', 'Clicks', 'Leads']
        target = ['Revenue']
        df_train = df_train[features + target]
        df_test = df_test[features + target]
        return df_train, df_test

    def save_data(self, 
                df: pd.DataFrame,
                output_path: str,
                name: str,
                version: str,
                ext: str) -> bool:
        return df.to_csv(f'{output_path}{name}_{version}{ext}', index=False)
        
    def run_step(self) -> bool:
        self.df_train, self.df_test = self.read_data(self.train_data_path, self.test_data_path)
        self.df_train, self.df_test = self.select_features(self.df_train, self.df_test)
        self.save_data(self.df_train, self.output_path, self.train_file_name, 'selected', self.train_file_ext)
        self.save_data(self.df_test, self.output_path, self.test_file_name, 'selected', self.test_file_ext)

        return True


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_data_path = config['storage']['feature_store_train_data']
    test_data_path = config['storage']['feature_store_test_data']
    output_path = config['storage']['selected_dir']

    select_features = SelectFeatures(
        train_data_path = train_data_path,
        test_data_path = test_data_path,
        output_path = output_path
    )
    select_features.run_step()
    logging.info('select_features script successfully completed run...')


if __name__ == "__main__":
    logging.info('Executing: select_features script..')
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()

    logging.info(f"Resolving configuration path: {config_path}")

    if not config_path.exists():
        logging.error(f"Configuration file missing: {config_path}")
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    logging.info(f'Successfully loaded configuration file at {config_path}')

    main(config_path)





# if __name__ == "__main__":
#     select_features = SelectFeatures(
#         train_data_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/feature_store/ads_campaign_data_train_created.csv',
#         test_data_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/feature_store/ads_campaign_data_test_created.csv',
#         output_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/selected/'
#     )
#     select_features.run_step()

