from typing import Tuple, Union, Optional

import pandas as pd
import numpy as np
from src.data_splitter import DataSplitter, KFoldSplitStrategy
import logging
from zenml import step


@step()
def data_splitter_step(
    X: pd.DataFrame, strategy: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets using DataSplitter and a chosen strategy."""

    # if strategy == 'train_test_split':
    #     splitter = DataSplitter(strategy=SimpleTrainTestSplitStrategy())
    #     X_train, X_test, y_train, y_test = splitter.split(df, target_column)
    #     return X_train, X_test, y_train, y_test

    if strategy == 'k_fold':
        splitter = DataSplitter(strategy=KFoldSplitStrategy())
        splits = splitter.split(X)

        # Ensure that we return the correct values
        # X_train, X_val, y_train_binary, y_val_binary, y_train_multi, y_val_multi, val_idx = splits
        X_train, X_val, y_train_binary, y_val_binary = splits

        # return X_train, X_val, y_train_binary, y_val_binary, y_train_multi, y_val_multi, val_idx
        return X_train, X_val, y_train_binary, y_val_binary
    else:
        raise ValueError(f"Unsupported data splitting strategy: {strategy}")
