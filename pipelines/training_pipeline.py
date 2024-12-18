from steps.data_ingestion_step import ingest_data
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitter_step import data_splitter_step
from steps.data_resampling_step import data_resampler_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluation_step
from zenml import Model, pipeline

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="predictive"
    ),
)

def ml_pipeline():
    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = ingest_data('predictive_maintenance')

    # Feature Engineering Step
    encoded_data = feature_engineering_step(
        raw_data, strategy="label_encoding", features=["Type", "Product ID", "Failure Type"]
    )

    # Data Splitting Step (K-Fold Cross-Validation)
    X_train, X_val, y_train_multiclass, y_val_multiclass = data_splitter_step(encoded_data, strategy='k_fold')

    # Data Resampling Step 
    X_train_multiclass_resampled, y_train_multiclass_resampled = data_resampler_step(X_train, y_train_multiclass)

    # Model Building Step 
    model = model_building_step(X_train_multiclass_resampled, y_train_multiclass_resampled)

     # Model Evaluator Step 
    f1_score = model_evaluation_step(model=model, X_val=X_val, y_val=y_val_multiclass
    )

    return model

if __name__ == "__main__":
    # Running the pipeline
    run = ml_pipeline()
