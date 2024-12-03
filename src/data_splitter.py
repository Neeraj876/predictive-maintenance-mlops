import logging
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from config import RANDOM_STATE, N_SPLITS

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Data Splitting Strategy
# -----------------------------------------------
# This class defines a common interface for different data splitting strategies.
# Subclasses must implement the split_data method.
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, X: pd.DataFrame):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.

        Returns:
        X_train, X_val, y_train_multi, y_val_multi: The training and validation splits for features and target for each fold.
        """
        pass
    
# Implement the Concrete Strategy for K-Fold Cross-Validation
# ---------------------------------------------
class KFoldSplitStrategy(DataSplittingStrategy):
    def __init__(self, n_splits=N_SPLITS, random_state=RANDOM_STATE):
        """
        Initializes the KFoldSplitStrategy with K-Fold parameters.

        Parameters:
        n_splits (int): Number of splits for K-Fold cross-validation.
        random_state (int): Random state for reproducibility.
        """
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split_data(self, X:pd.DataFrame):
        """
        Generator that yields train and validation splits for each fold using K-Fold.

        Parameters:
        X (pd.DataFrame): Label Encoded Dataframe. 

        Yields:
        X_train, X_val, y_train_multi, y_val_multi: The training and validation splits for features and target for each fold.
        """
    
        # splits = []
        # for fold, (train_idx, val_idx) in enumerate(self.kf.split(X)):
        #     # Split features
        #     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        #     # Split binary targets
        #     y_train_binary, y_val_binary = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
        #     # Split multiclass targets
        #     y_train_multi, y_val_multi = y_multiclass.iloc[train_idx], y_multiclass.iloc[val_idx]
            
        #     # Store splits in a list
        #     splits.append((X_train, X_val, y_train_binary, y_val_binary, y_train_multi, y_val_multi, val_idx))

        # # Return all splits
        # return splits

        print("Encoded_data is: ", X)

        logging.info("Performing K-Fold cross-validation split.")
        y_multiclass = X['Failure Type_encoded']
        X = X.drop(columns=['Target', 'Failure Type_encoded', 'id'])

        for fold, (train_idx, val_idx) in enumerate(self.kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_multi, y_val_multi = y_multiclass.iloc[train_idx], y_multiclass.iloc[val_idx]

            print('X_train labels', X_train.columns)
            print('X_train is', X_train)
            logging.info("K-Fold cross-validation split completed.")
            return X_train, X_val, y_train_multi, y_val_multi

# Context Class for Data Splitting
# --------------------------------
# This class uses a DataSplittingStrategy to split the data.
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Switching data splitting strategy.")
        self._strategy = strategy

    def split(self, X: pd.DataFrame):
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.

        Returns:
        X_train, X_val, y_train_multi, y_val_multi: The training and testing splits for features and target.
        """
        return self._strategy.split_data(X)

# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    # df = pd.read_csv('/mnt/c/Users/HP/ml_projects/predictive_maintenance/extracted_data/predictive_maintenance.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    # df = pd.DataFrame({
    # 'feature1': np.random.rand(100),
    # 'feature2': np.random.rand(100),
    # 'SalePrice': np.random.randint(100000, 500000, 100)
    # })
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')
    # print("Simple Split:")
    # print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # data_splitter = DataSplitter(KFoldSplitStrategy(n_splits=5, random_state=42))

    # X_train, X_val, y_train_binary, y_val_binary, y_train_multi, y_val_multi, val_idx = data_splitter.split(df)
    # print(f"X shape: {df.shape}")
    # print(f"y_bin shape: {y_bin.shape}")
    # print(f"y_multiclass shape: {y_multiclass.shape}")
    # print(f"First 5 values of y_bin: {y_bin.head()}")
    # print(f"First 5 values of y_multiclass: {y_multiclass.head()}")
    # print(df.columns)
    pass

