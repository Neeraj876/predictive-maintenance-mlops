from typing import Tuple
import logging
import pandas as pd
from src.data_resampling import DataResampler, OversamplingStrategy
from zenml import step

@step()
def data_resampler_step(
    X_train: pd.DataFrame, 
    # y_train_binary: pd.Series, 
    y_train_multiclass: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample the data using SMOTE and Undersampling using DataResampler and a chosen strategy.

    Parameters:
    - X_train: The training features.
    - y_train_binary: The binary target variable.
    - y_train_multiclass: The multiclass target variable.

    Returns:
    - X_train_resampled: The resampled training features.
    - y_train_binary_resampled: The resampled binary target variable.
    - y_train_multiclass_resampled: The resampled multiclass target variable.
    """
    logging.info("Starting resampling step...")

    # Initialize the resampler with the specified strategy
    resampler = DataResampler(strategy=OversamplingStrategy())

    # Perform the resampling
    X_train_multiclass_resampled, y_train_multiclass_resampled = resampler.apply_resampling(X_train, y_train_multiclass)

    # Logging the shapes of the resampled data for validation
    logging.info(f"Resampled data shapes: X_train_multiclass_resampled={X_train_multiclass_resampled.shape},  y_train_multiclass_resampled={y_train_multiclass_resampled.shape}")

    logging.info(f"Resampled data value counts:  y_train_multiclass_resampled={y_train_multiclass_resampled.value_counts()}")


    # Return the resampled data
    print('X_train_multiclass_resampled labels after resampling', X_train_multiclass_resampled.columns)
    return X_train_multiclass_resampled,  y_train_multiclass_resampled
  
