from typing import Tuple
import logging
import pandas as pd
from src.data_resampling import DataResampler, OversamplingStrategy
from zenml import step

@step()
def data_resampler_step(
    X_train: pd.DataFrame, 
    y_train_multiclass: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Resample the data using DataResampler and a chosen strategy.
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
  
