from steps.data_ingestion_step import data_ingestion_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluation_step
from zenml import Model, pipeline, step

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="predictive"
    ),
)

def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = data_ingestion_step(
        file_path="/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/data/archive.zip"
    )

    # Feature Engineering Step
    encoded_data = feature_engineering_step(
        raw_data, strategy="label_encoding", features=["Type", "Product ID", "Failure Type"]
    )

    # Data Splitting Step (K-Fold Cross-Validation)
    X_train, X_val, y_train_binary, y_val_binary= data_splitter_step(encoded_data, strategy='k_fold')

    # Model Building Step (Binary Classification)
    model = model_building_step(X_train=X_train, y_train=y_train_binary, method='binary')

    # Model Evaluator Step 
    f1_score = model_evaluation_step(model=model, X_val=X_val, y_val=y_val_binary
    )

    return model



if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
