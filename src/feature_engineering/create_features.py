import os
import sys
from pathlib import Path
from typing import Tuple
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

class CreateFeatures:
    """ Creating features based on Exploration """

    def __init__(self, train_data_path:str, test_data_path:str, output_path:str) -> None:
        """
        Args:
            train_data_path (str): Path of the preprocessed train data
            test_data_path (str): Path of the preprocessed test data
            output_path (str): Where to save the new dataframe to
        Returns:
            None
        """
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
        """
        Read data using pandas or spark

        Args:
            train_data_path (str): Path of the preprocessed train data
            test_data_path (str): Path of the preprocessed test data
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: the dataframes read from the path
        """
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        return df_train, df_test

    def create_features(self, df_train:pd.DataFrame, df_test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Creating high quality features 
        
        Args:
            df_train (pd.DataFrame): train data read from the path
            df_test (pd.DataFrame): test data read from the path
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: the dataframes with created features
        """
        df_train['Clicks_Plus_Leads'] = df_train['Clicks'] + df_train['Leads']
        df_test['Clicks_Plus_Leads'] = df_test['Clicks'] + df_test['Leads']
        return df_train, df_test

    def save_data(self, 
                df: pd.DataFrame,
                output_path: str,
                name: str,
                version: str,
                ext: str) -> bool:
        """
        Args:
            df (pd.DataFrame): the dataframe to save
            output_path (str): the path to save the df
            name (str): the name to use in saving the file
            version (str): version of the file
            ext (str): file extension
        Returns:
            (bool): whether the dataframe was saved or not
        """
        return df.to_csv(f'{output_path}{name}_{version}{ext}', index=False)
        
    def run_step(self) -> bool:
        """
        Runs the class step in the expected order

        Args:
            None
        Returns:
            (bool): confirmation of success
        """
        self.df_train, self.df_test = self.read_data(self.train_data_path, self.test_data_path)
        self.df_train, self.df_test = self.create_features(self.df_train, self.df_test)
        self.save_data(self.df_train, self.output_path, self.train_file_name, 'created', self.train_file_ext)
        self.save_data(self.df_test, self.output_path, self.test_file_name, 'created', self.test_file_ext)

        return True


def main(config_path: str):
    """
    Calls the CreateFeatures class

    Args:
        config_path (str): path to the configuration file for the run
    Returns:
        (bool): confirmation of success
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_data_path = config['storage']['processed_train_data']
    test_data_path = config['storage']['processed_test_data']
    output_path = config['storage']['feature_store_dir']

    create_features = CreateFeatures(
        train_data_path = train_data_path,
        test_data_path = test_data_path,
        output_path = output_path
    )
    create_features.run_step()
    logging.info('create_features script successfully completed run...')


if __name__ == "__main__":
    logging.info('Executing: create_features script..')
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


