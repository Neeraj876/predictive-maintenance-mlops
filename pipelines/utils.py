import logging
import pandas as pd
import re

from src.feature_engineering import FeatureEngineer, labelEncoding 



def get_data_for_test():
    try:
        df = pd.read_csv('/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/extracted_data/predictive_maintenance.csv')
        df = df.sample(n=100)
        label_encoding = labelEncoding(features=["Type", "Product ID", "Failure Type"])
        encoded_data = FeatureEngineer(label_encoding)
        df = encoded_data.apply_feature_engineering(df)
        print('Data after label_encoding is', df)
        df.drop(['Target', 'Failure Type_encoded'], axis=1, inplace=True)
        df.columns = [re.sub(r"[<>[\]]", "", col) for col in df.columns]
        print("Cleaned column names:", df.columns)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e

    