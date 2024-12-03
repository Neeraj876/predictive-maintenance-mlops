import mlflow
import logging
import pandas as pd
from typing import Annotated
from sklearn.base import ClassifierMixin
from src.model_building import (
    MulticlassModelTrainingStrategy,
    ModelBuilder
)
from materializers.custom_materializer import XGBClassifierMaterializer

from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

if experiment_tracker is None:
    raise ValueError("Experiment tracker is not initialized. Please ensure ZenML is set up correctly.")

model = Model(
    name="predictive_maintenance",
    version=None,
    license="Apache 2.0",
    description="Predictive maintenance model for machines.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[ClassifierMixin, ArtifactConfig(name="new_model", materializer=XGBClassifierMaterializer)]:
    """
    Builds and trains a Classification model.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train_multiclass must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train_multiclas must be a pandas Series.")

    # Start an MLflow run to log the training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging to automatically log model parameters, metrics, and artifacts
        mlflow.sklearn.autolog()

        # Log training data (optional but useful for tracking)
        mlflow.log_param("X_train_multiclass shape", X_train.shape)
        mlflow.log_param("y_train_multiclass shape", y_train.shape)

        logging.info("Building and training the multiclass classification model.")
     
        # Train multiclass classification model
        model_builder = ModelBuilder(MulticlassModelTrainingStrategy())
        logging.info("Building and training the multiclass classification model.")
        trained_model = model_builder.build_model(X_train=X_train, y_train=y_train)
        logging.info("Multiclass classification model training completed.")

        # Explicitly log the trained model
        mlflow.sklearn.log_model(trained_model, "model", input_example=X_train.iloc[:1])
        logging.info("Multiclass Model logged successfully in MLflow.")

        logging.info("Model training completed.")

    except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise e
    finally:
        # End the MLflow run
        mlflow.end_run()

    return trained_model