import mlflow
import logging
import pandas as pd
import numpy as np
from typing import List, Annotated, Tuple, Dict
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
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """
    # Ensure the inputs are of the correct type
    # if not isinstance(X_train_binary, pd.DataFrame):
    #     raise TypeError("X_train_binary must be a pandas DataFrame.")
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train_multiclass must be a pandas DataFrame.")
    # if not isinstance(y_train_binary, pd.Series):
    #     raise TypeError("y_train_binary must be a pandas Series.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train_multiclas must be a pandas Series.")
     
    # logging.info(f"Building model using method: {method}")
    
    # if method == "logistic_regression":
    #     strategy = LogisticRegressionStrategy()
    #     logging.info("Selected Logistic Regression Strategy.")
    
    # elif method == "binary":
    #     strategy = BinaryModelTrainingStrategy()
    #     logging.info("Selected Binary Classification Strategy.")

    # elif method == "multi":
    #     strategy = MulticlassModelTrainingStrategy()
    #     logging.info("Selected Multiclass Classification Strategy.")

    # elif method == "svc":
    #     strategy = SVCStrategy()
    #     logging.info("Selected SVM Strategy.")

    # elif method == "naive_bayes":
    #     strategy = NaiveBayesStrategy()
    #     logging.info("Selected Naive Bayes Strategy.")

    # elif method == "random_forest":
    #     strategy = RandomForestStrategy()
    #     logging.info("Selected Random Forest Strategy.")

    # else:
    #     raise ValueError(f"Unknown method '{method}' selected for model training.")
    
    # Initialize ModelBuilder with the selected strategy
    # model_builder = ModelBuilder()

    # Start an MLflow run to log the training process
    if not mlflow.active_run():
        mlflow.start_run()

    try:
        # Enable autologging to automatically log model parameters, metrics, and artifacts
        mlflow.sklearn.autolog()

        # Log training data (optional but useful for tracking)
        # mlflow.log_param("X_train_binary shape", X_train_binary.shape)
        mlflow.log_param("X_train_multiclass shape", X_train.shape)
        # mlflow.log_param("y_train_binary shape", y_train_binary.shape)
        mlflow.log_param("y_train_multiclass shape", y_train.shape)

        logging.info("Building and training the multiclass classification model.")
     
        # Train multiclass classification model
        model_builder = ModelBuilder(MulticlassModelTrainingStrategy())
        logging.info("Building and training the multiclass classification model.")
        trained_model = model_builder.build_model(X_train=X_train, y_train=y_train)
        logging.info("Multiclass classification model training completed.")

        # Explicitly log the trained model
        mlflow.sklearn.log_model(trained_model, "model", input_example=X_train.iloc[:1])
        logging.info("Binary Model logged successfully in MLflow.")
        
        # logging.info("Building and training the multi classification model.")

        # # Train multiclass classification model
        # multiclass_model = ModelBuilder(MulticlassModelTrainingStrategy())
        # trained_multiclass_model = multiclass_model.build_model(X_train=X_train_multiclass, y_train=y_train_multiclass)
        # logging.info("Multiclass classification model training completed.")

        # # Explicitly log the trained model
        # mlflow.sklearn.log_model(trained_multiclass_model, "multiclass_model", input_example=X_train_multiclass.iloc[:1])
        # logging.info("Multiclass Model logged successfully in MLflow.")

        logging.info("Model training completed.")

    except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise e
    finally:
        # End the MLflow run
        mlflow.end_run()

    return trained_model