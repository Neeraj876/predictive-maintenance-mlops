import logging

import pandas as pd

from zenml import step

from src.ingest_data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@step(enable_cache=False)
def ingest_data(
    table_name: str , 
) -> pd.DataFrame:
    """Reads data from sql database and return a pandas dataframe.

    Parameters:
        table_name : str
        The name of the table from which to load data.

    Args:
        data: pd.DataFrame
    """

    logging.info("Started data ingestion process.")

    try:
        data_loader = DataLoader('postgresql://postgres:neeraj@localhost:5432/predictive_pg')
        data_loader.load_data(table_name) 
        df = data_loader.get_data()  
        if df.empty:
            logging.warning("No data was loaded. Check the table name or the database content.")
        else:
            logging.info(f"Data ingestion completed. Number of records loaded: {len(df)}.")
    
        logging.info("Data loaded successfully")

        return df  
    except Exception as e:
        logging.error(f"Error while reading data from {table_name}: {e}")
        raise e