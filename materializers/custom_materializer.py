import os
import pickle
import numpy as np
from typing import Any, Type, Union
import xgboost as xgb
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

DEFAULT_FILENAME_XGB = "xgb.pkl"  # For the XGBClassifier model

class XGBClassifierMaterializer(BaseMaterializer):
    """
    Custom materializer for XGBClassifier model
    """

    ASSOCIATED_TYPES = (
        xgb.XGBClassifier, 
    )

    def handle_input(self, data_type: Type[Any]) -> xgb.XGBClassifier:
        """
        Loads the XGBoost model from the artifact and returns it.

        Args:
            data_type: The type of the model to be loaded
        """
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME_XGB)

        try:
            with fileio.open(filepath, "rb") as fid:
                model = pickle.load(fid)
            return model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {filepath}. Ensure the file is present in the artifact store.") from e
        except Exception as e:
            raise RuntimeError(f"Error loading the XGBoost model from {filepath}: {e}") from e

    def handle_return(self, obj: xgb.XGBClassifier) -> None:
        """
        Saves the XGBoost model to the artifact store.

        Args:
            model: The XGBClassifier model to be saved
        """
        super().handle_return(obj)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME_XGB)

        try:
            with fileio.open(filepath, "wb") as fid:
                pickle.dump(obj, fid)
        except Exception as e:
            raise RuntimeError(f"Error saving the XGBoost model to {filepath}: {e}") from e
        