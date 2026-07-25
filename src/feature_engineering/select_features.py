import os
import sys
from pathlib import Path
from typing import Tuple, List
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
from src.utils.logger import get_pipeline_logger


class SelectFeatures:
    """ Feature selection based on Exploration """
    
    def __init__(
                self,
                train_data_path:str,
                test_data_path:str,
                output_path:str,
                selected_features:List,
                target_column:str,
                logging:logging.Logger) -> None:
        """
        Args:
            train_data_path (str): Path of the create_features train data
            test_data_path (str): Path of the create_features test data
            output_path (str): Where to save the new dataframe to
        Returns:
            None
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.output_path = output_path
        self.selected_features = selected_features
        self.target_column = target_column
        self.logging = logging
        
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
            train_data_path (str): Path of the create_features train data
            test_data_path (str): Path of the create_features test data
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: the dataframes read from the path
        """
        df_train = pd.read_csv(train_data_path)
        df_test = pd.read_csv(test_data_path)
        return df_train, df_test

    def select_features(self, df_train:pd.DataFrame, df_test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Selecting high quality features 
        
        Args:
            df_train (pd.DataFrame): train data read from the path
            df_test (pd.DataFrame): test data read from the path
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: the dataframes with selected features
        """
        features = self.selected_features
        target = self.target_column
        df_train = df_train[features + target]
        df_test = df_test[features + target]
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
        - Creates the select_features output folder
        - Runs the class step in the expected order

        Args:
            None
        Returns:
            (bool): confirmation of success
        """
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.df_train, self.df_test = self.read_data(self.train_data_path, self.test_data_path)
        self.df_train, self.df_test = self.select_features(self.df_train, self.df_test)
        self.save_data(self.df_train, self.output_path, self.train_file_name, 'selected', self.train_file_ext)
        self.save_data(self.df_test, self.output_path, self.test_file_name, 'selected', self.test_file_ext)
        self.logging.info('Process complete!')

        return True


def main(config_path: str):
    """
    Calls the SelectFeatures class

    Args:
        config_path (str): path to the configuration file for the run
    Returns:
        (bool): confirmation of success
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_output_top_level = config['run_output']['top_level']
    run_output_name = config['name']
    run_output = run_output_top_level + run_output_name

    log_dir = Path(run_output + "/logs")
    log_file = log_dir / "production.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_pipeline_logger(log_file_path=log_file)
    logger.info("Running select_features...")

    train_data_path = run_output + config['run_output']['feature_store_train_data']
    test_data_path = run_output + config['run_output']['feature_store_test_data']
    output_path = run_output + config['run_output']['selected_dir']
    selected_features = config['feature_selection']['selected_features']
    target_column = config['feature_selection']['target_column']

    select_features = SelectFeatures(
        train_data_path = train_data_path,
        test_data_path = test_data_path,
        output_path = output_path,
        selected_features = selected_features,
        target_column = target_column,
        logging = logger
    )
    select_features.run_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    main(config_path)
