import logging
from abc import ABC, abstractmethod
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.utils import shuffle

from typing import Tuple

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the Abstract Base Class for Data Preprocessing Strategy
# -----------------------------------------------
# This class defines a common interface for data preprocessing strategy.
# Subclasses must implement the resample method.
class DataResamplingStrategy(ABC):
    @abstractmethod
    def resample(self, X_train: pd.DataFrame, y_train_multiclass: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method to apply SMOTE and undersampling for two dependent variables: `target` and `failure_type` in training set.

        Parameters:
        X_train (pd.DataFrame): The input DataFrame to be resampled.
        y_train_binary (pd.Series): First target column.
        y_train_multiclass (pd.Series): Second target column 

        Returns:
        X_train_resampled : pd.DataFrame
            Resampled feature data.
        y_train_binary_resampled : pd.Series
            Resampled first target column.
        y_train_multiclass_resampled : pd.Series
            Resampled second target column.
        """
        pass

# Implements the Concrete Strategy for SMOTE and undersampling resampling techniques 
# ---------------------------------------------
# This strategy implements the SMOTE and undersampling resampling techniques.

class OversamplingStrategy(DataResamplingStrategy):
    def __init__(self, random_state: int = 42):
        """
        Initializes the SmoteAndUndersamplingStrategy with specific parameters.

        Parameters:
        smote_strategy (float, default=0.5): SMOTE strategy for combined target variable.
        undersample_strategy (float, default=0.5):
            Sampling strategy for RandomUnderSampler.
        random_state (int, default=42):
            Random state for reproducibility.
        """
        self.random_state = random_state

    def resample(self, X_train: pd.DataFrame, y_train_multiclass: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply oversampling for two dependent variables: `target` and `failure_type`.

        Parameters:
        X_train : pd.DataFrame
            Feature data.
        y_train_binary : pd.Series
            First target column.
        y_train_multiclass : pd.Series
            Second target column 

        Returns:
        X_train_binary_resampled : pd.DataFrame
            Resampled X_train feature data for binary target variable.
        X_train_multiclass_resampled : pd.DataFrame
            Resampled  X_train feature data for binary multiclass variable.
        y_target_resampled : pd.Series
            Resampled first target column.
        y_failure_type_resampled : pd.Series
            Resampled second target column.
        """
        # Apply SMOTETomek for the binary target (Target)
        # smote_tomek_target = SMOTETomek(random_state=42)
        # X_train_binary_resampled, y_train_binary_resampled = smote_tomek_target.fit_resample(X_train, y_train_binary)

        # Apply SMOTETomek for the multi-class target (Failure Type)
        smote_tomek_failure = SMOTETomek(random_state=42)
        X_train_multiclass_resampled, y_train_multiclass_resampled = smote_tomek_failure.fit_resample(X_train, y_train_multiclass)
            
        return X_train_multiclass_resampled,y_train_multiclass_resampled
    
# Context Class for Data Resampling
# --------------------------------
# This class uses a DataResamplingStrategy to resample the data.
class DataResampler:
    def __init__(self, strategy: DataResamplingStrategy):
        """
        Initializes the DataResampler with a specific data resampling strategy.

        Parameters:
        strategy (DataResamplingStrategy): The strategy to be used for data resampling.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataResamplingStrategy):
        """
        Sets a new strategy for the DataResampler.

        Parameters:
        strategy (DataResamplingStrategy): The new strategy to be used for data resampling.
        """
        logging.info("Switching data resampling strategy.")
        self._strategy = strategy

    def apply_resampling(self, X_train: pd.DataFrame, y_train_multiclass: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Executes the data resampling using the current strategy.

        Parameters:
        X_train : pd.DataFrame
            Feature data.
        y_train_binary : pd.Series
            First target column.
        y_train_multiclass : pd.Series
            Second target column

        Returns:
        X_train_resampled : pd.DataFrame
            Resampled feature data.
        y_train_binary_resampled : pd.Series
            Resampled first target column.
        y_train_multiclass_resampled : pd.Series
            Resampled second target column.
        """
        # logging.info(f"X_train shape: {X_train.shape}")
        # # logging.info(f"y_train_binary shape: {y_train_binary.shape}")
        # logging.info(f"y_train_multiclass shape: {y_train_multiclass.shape}")

        return self._strategy.resample(X_train, y_train_multiclass)

if __name__ == "__main__":
    # Example usage
    # Create a sample imbalanced dataset
    # data = pd.DataFrame({
    #     'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
    #     'feature2': [5, 6, 7, 8, 9, 10, 11, 12],
    #     'target': [0, 0, 0, 0, 1, 0, 1, 1],
    #     'failure_type': ['A', 'A', 'A', 'B', 'B', 'A', 'B', 'C']
    # })
    
    # logging.info("Starting preprocessing for the sample dataset.")
    # feature_columns = ['feature1', 'feature2']
    # target_columns = ['target', 'failure_type']
    
    # X, y_target, y_failure_type = preprocess_features_and_targets(data, feature_columns, target_columns)
    
    # # Apply SMOTE and undersampling
    # X_resampled, y_target_resampled, y_failure_type_resampled = apply_smote_and_undersampling_multioutput(
    #     X, y_target, y_failure_type, smote_strategy_target=0.8, smote_strategy_failure_type=0.8, undersample_strategy=0.7
    # )
    
    # logging.info("Preprocessing complete.")
    # print("Original Data:\n", data)
    # print("\nResampled Data Features:\n", X_resampled)
    # print("\nResampled Data `target`:\n", y_target_resampled)
    # print("\nResampled Data `failure_type`:\n", y_failure_type_resampled)
    pass
