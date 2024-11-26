# config.py

# Configuration file for storing constants and parameters

# File paths
DATA_PATH = "/mnt/c/Users/HP/ml_projects/predictive_maintenance/extracted_data/predictive_maintenance.csv"

# Random states for reproducibility
RANDOM_STATE = 42

# Feature selection threshold (mean mutual information score)
MI_SCORE_THRESHOLD = None  # To be set dynamically after calculation

# Columns
CATEGORICAL_COLS = ["Type", "Product ID", "Failure Type"]
NUMERICAL_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# K-Fold configuration
N_SPLITS = 5
