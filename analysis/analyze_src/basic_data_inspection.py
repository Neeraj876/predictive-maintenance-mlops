from abc import ABC, abstractmethod

import pandas as pd

# Define The Abstract Base Class or A Common Interface for Data Inspection Strategies

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """Perform a specific type of data inspection.
        
        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly
        """
        pass

# Implement the concrete class/strategy for Data Types Inspection
# ------------------------------------------------------
# This strategy inspects the data types of each column and count non-null values

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\n Data Types and Non-null Counts:")
        print(df.info())

# Implement the concrete class/strategy for summary statistics inspection
# ------------------------------------------------------------
# This strategy provides summary statistics for both numerical and categorical features.

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for both numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns: 
        None: Prints the summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features):")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(df.describe(include=["O"]))

# Implement the Context class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows to switch between different data inspection strategies
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def  set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    
    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the current strategy's inspection method.
        """
        self._strategy.inspect(df)
    
# Example Usage
if __name__ == "__main__":
    # Load the Data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass