import json
import logging
import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters
# from pipelines.training_pipeline import ml_pipeline


from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.data_resampling_step import data_resampler_step
from steps.data_ingestion_step import ingest_data
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluation_step

from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float = 0

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    print('Data from utils', data)
    return data

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Implements a simple model deployment trigger that looks at the input model accuracy and decides if it is good enough to deploy or not"""
    return accuracy > config.min_accuracy

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str="model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """    
    try:
        # get the MLflow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # fetch existing services with same pipeline name, step name and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name=pipeline_name,
            pipeline_step_name=pipeline_step_name,
            model_name=model_name,
            running=running,
        )

        if not existing_services:
            raise RuntimeError(
                f"No MLflow prediction service deployed by the "
                f"{pipeline_step_name} step in the {pipeline_name} "
                f"pipeline for the '{model_name}' model is currently "
                f"running."
            )
        return existing_services[0]
    except Exception as e:
        logging.error(f"Error in prediction_service_loader: {str(e)}")
        raise

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)

    # Check for required keys in the data
    if "data" not in data or "columns" not in data or "index" not in data:
        logging.error("Input data does not have the required keys: 'data', 'columns', 'index'")
        raise KeyError("Input data is missing required keys")
    data.pop("columns")
    data.pop("index")
    columns_for_df =[
        'UDI', 
        'Air temperature K', 
        'Process temperature K', 
        'Rotational speed rpm', 
        'Torque Nm', 
        'Tool wear min', 
        'Type_encoded', 
        'Product ID_encoded'
    ]

    # Check if data has the correct number of features
    if len(data["data"][0]) != len(columns_for_df):
        logging.error(f"Input data has incorrect number of features: expected {len(columns_for_df)}, got {len(data['data'][0])}")
        raise ValueError("Input data has incorrect number of features")

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 3,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    raw_df = ingest_data(data_path)
    encoded_df = feature_engineering_step(raw_df, strategy="label_encoding", features=["Type", "Product ID", "Failure Type"])
    X_train, X_val, y_train_multiclass, y_val_multiclass =  data_splitter_step(encoded_df, strategy='k_fold')
    X_train_multiclass_resampled, y_train_multiclass_resampled = data_resampler_step(X_train, y_train_multiclass)
    model = model_building_step(X_train=X_train_multiclass_resampled, y_train=y_train_multiclass_resampled)
    f1_score = model_evaluation_step(model=model, X_val=X_val, y_val=y_val_multiclass)
    deployment_decision = deployment_trigger(f1_score)

    # Run the training pipeline
    # trained_model = ml_pipeline()

    # Deploy the trained model
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=model_deployment_service, data=batch_data)
    return prediction