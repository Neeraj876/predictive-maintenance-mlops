import logging
from abc import ABC, abstractmethod
from imblearn.combine import SMOTETomek
import pandas as pd
from sklearn.utils import shuffle

from typing import Tuple

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the Abstract Base Class for Data Resampling Strategy
# -----------------------------------------------
# This class defines a common interface for data resampling strategy.
# Subclasses must implement the resample method.
class DataResamplingStrategy(ABC):
    @abstractmethod
    def resample(self, X_train: pd.DataFrame, y_train_multiclass: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Abstract method to apply oversampling for target variable `failure_type` in training set.

        Parameters:
        X_train (pd.DataFrame): The input DataFrame to be resampled.
        y_train_multiclass (pd.Series): target column 

        Returns:
        X_train_multiclass_resampled (pd.DataFrame):
            Resampled feature data.
        y_failure_type_resampled (pd.Series):
            Resampled target column.
        """
        pass

# Implements the Concrete Strategy for oversampling resampling techniques 
# ---------------------------------------------
# This strategy implements the oversampling resampling techniques.

class OversamplingStrategy(DataResamplingStrategy):
    def __init__(self, random_state: int = 42):
        """
        Initializes the OversamplingStrategy with specific parameters.

        Parameters:
        random_state (int, default=42):
            Random state for reproducibility.
        """
        self.random_state = random_state

    def resample(self, X_train: pd.DataFrame, y_train_multiclass: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply oversampling for a target variable `failure_type`.

        Parameters:
        X_train : pd.DataFrame
            Feature data
        y_train_multiclass : pd.Series
            target column

        Returns:
        X_train_multiclass_resampled (pd.DataFrame):
            Resampled feature data.
        y_failure_type_resampled (pd.Series):
            Resampled target column.
        """
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
        return self._strategy.resample(X_train, y_train_multiclass)

if __name__ == "__main__":
    pass
