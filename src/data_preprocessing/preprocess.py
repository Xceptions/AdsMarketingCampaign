import os
from pathlib import Path
import numpy as np
import pandas as pd


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
        """Preprocessing the data using pandas or spark"""
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

        return True

if __name__ == "__main__":
    preprocess = Preprocess(
        data_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/raw/ads_campaign_data.csv',
        output_path = '/Users/macbookair/Documents/GitHub/AdsMarketingCampaign/data/processed/'
    )
    preprocess.run_step()

