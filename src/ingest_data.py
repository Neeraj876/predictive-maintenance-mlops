import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd

# Define the abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

# Implement a concrete class for Zip Ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path):
        """Extract a .zip file and return the content as a pandas DataFrame"""
        # Ensure the file is .zip
        if not file_path.endswith('.zip'):
            raise ValueError('The provided file is not a .zip file')
        
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall('extracted_data')

        # Find the extracted CSV file (assuming there is one CSV file inside the zip)
        extracted_files = os.listdir('extracted_data')
        csv_files = [f for f in extracted_files if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            raise FileNotFoundError('No CSV file found in the extracted data.')
        elif len(csv_files) > 1:
            raise FileNotFoundError('Multiple CSV files are found. Please specify which one to use')
        
        # Read the CSV file into a DataFrame
        csv_file_path = os.path.join('extracted_data', csv_files[0])
        df = pd.read_csv(csv_file_path)

        # Return the DataFrame
        return df
    
# Implement a factory class to create Data Ingestors object
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> pd.DataFrame:
        if file_extension == 'zip':
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension {file_extension}") 
    
# Example usage    
if __name__ == "__main__":

    # # Specify the file path
    #file_path = '/mnt/c/Users/HP/ml_projects/predictive_maintenance/data/archive.zip'

    # # Determine the file extension
    #file_extension = os.path.splitext(file_path)[1][1:]
    #print(file_extension)

    # # Get the appropriate DataIngestor
    #data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    # # Ingest the data and load it into a DataFrame
    #df = data_ingestor.ingest(file_path)

    # # Now df contains the DataFrame from the extracted CSV
    #print(df.head())
    pass



