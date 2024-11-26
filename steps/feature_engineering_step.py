import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScaling,
    OneHotEncoding,
    StandardScaling,
    labelEncoding, 
)

from zenml import step

@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str, features: list = None
) -> pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and selected strategy."""

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaling(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaling(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoding(features))
    elif strategy == 'label_encoding':
        engineer = FeatureEngineer(labelEncoding(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)

    print('Feature labels', transformed_df.columns)
    return transformed_df


