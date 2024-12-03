import logging
import pandas as pd
import re

from src.ingest_data import DataLoader
from src.feature_engineering import FeatureEngineer, labelEncoding 

from config import db_uri

def get_data_for_test():
    try:
        logging.info("Loading data from PostgreSQL table.")
        data_loader = DataLoader(db_uri)
        data_loader.load_data('predictive_maintenance')
        df = data_loader.get_data()

        if df.empty:
            logging.warning("Loaded data is empty. Check database table content.")

        logging.info(f"Data loaded successfully with {len(df)} records.")
        df = df.sample(n=100)
        label_encoding = labelEncoding(features=["Type", "Product ID", "Failure Type"])
        encoded_data = FeatureEngineer(label_encoding)
        df = encoded_data.apply_feature_engineering(df)
        print('Data after label_encoding is', df)
        df.drop(['Target', 'Failure Type_encoded', 'id'], axis=1, inplace=True)
        df.columns = [re.sub(r"[<>[\]]", "", col) for col in df.columns]
        print("Cleaned column names:", df.columns)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e

    