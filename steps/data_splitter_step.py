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

        X_train, X_val, y_train_multi, y_val_multi = splits

        #return X_train, X_val, y_train_binary, y_val_binary

        # Extract the first fold (for simplicity; modify if you want to use all folds)
        # X_train, X_val, y_train_binary, y_val_binary, y_train_multi, y_val_multi = next(splits)

        # Ensure that we return the correct values
        return X_train, X_val, y_train_multi, y_val_multi
    else:
        raise ValueError(f"Unsupported data splitting strategy: {strategy}")
