import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


class CreateFeatures:
    """ Feature creation based on Exploration """
    
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

    def create_features(self, df_train:pd.DataFrame, df_test:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Creating high quality features """
        df_train['Clicks_Plus_Leads'] = df_train['Clicks'] + df_train['Leads']
        df_test['Clicks_Plus_Leads'] = df_test['Clicks'] + df_test['Leads']
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
        self.df_train, self.df_test = self.create_features(self.df_train, self.df_test)
        self.save_data(self.df_train, self.output_path, self.train_file_name, 'created', self.train_file_ext)
        self.save_data(self.df_test, self.output_path, self.test_file_name, 'created', self.test_file_ext)

        return True

if __name__ == "__main__":
    create_features = CreateFeatures(
        train_data_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/processed/ads_campaign_data_train.csv',
        test_data_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/processed/ads_campaign_data_test.csv',
        output_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/feature_store/'
    )
    create_features.run_step()

