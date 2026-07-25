import os
import sys
from pathlib import Path
import yaml
import argparse
import logging
import numpy as np
import pandas as pd
from src.utils.logger import get_pipeline_logger


class Preprocess:
    """ Preprocessing based on Exploration """
    
    def __init__(self, data_path:str, output_path:str, logging:logging.Logger) -> None:
        """
        Args:
            data_path (str): Path of the raw data
            output_path (str): Where to save the preprocessed data to
        Returns:
            None
        """
        self.data_path = data_path
        self.output_path = output_path
        self.file_path = Path(data_path)
        self.file_name = self.file_path.stem
        self.file_ext = self.file_path.suffix
        self.logging = logging

    def read_data(self, data_path: str):
        """
        Read data using pandas or spark

        Args:
            data_path (str): Path of the raw data
        Returns:
            pd.DataFrame: the dataframe read from the path
        """
        df = pd.read_csv(data_path)
        return df

    def preprocess(self, df: pd.DataFrame) -> None:
        """
        Preprocessing the data using pandas or spark.
        Split into train-test, cleaning, imputation

        Args:
            df (pd.DataFrame): the dataframe read from the path
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: a preprocessed split of
                                               train and test data
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
        - Creates the preprocess output directory
        - Runs the preprocess step in the expected order

        Args:
            None
        Returns:
            (bool): confirmation of success
        """
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.df = self.read_data(self.data_path)
        self.df_train, self.df_test = self.preprocess(self.df)
        self.save_data(self.df_train, self.output_path, self.file_name, 'train', self.file_ext)
        self.save_data(self.df_test, self.output_path, self.file_name, 'test', self.file_ext)
        self.logging.info(f'Process complete')

        return True

def main(config_path: str):
    """
    Calls the Proprocess class

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
    logger.info("Running preprocessing...")

    data_path = config['storage']['base_data']
    output_path = run_output + config['run_output']['processed_dir']

    preprocess = Preprocess(
        data_path = data_path,
        output_path = output_path,
        logging = logger
    )
    preprocess.run_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    main(config_path)