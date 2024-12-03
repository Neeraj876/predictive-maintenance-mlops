import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
import re

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Evaluation Strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, model:ClassifierMixin, X_val:pd.DataFrame, y_val:pd.Series
    ) -> Any:
        """
        Abstract method to evaluate a model.

        Parameters:
            model (ClassifierMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Any: A dictionary containing evaluation metrics or a single evaluation metric.
        """
        pass

# Concrete Strategy for Classification Model Evaluation
class ClassificationModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: ClassifierMixin, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """
        Evaluates a classification model using various metrics.

        Parameters:
            model (ClassifierMixin): The trained classification model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Any: A dictionary containing accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix or a single evaluation metric.
        """

        logging.info("Cleaning column names in X_val.")
        # Clean column names in X_val to avoid special characters issues
        X_val.columns = [re.sub(r"[<>[\]]", "", col) for col in X_val.columns]
        
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_val)
        
        logging.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average="weighted")  # Adjust average as needed
        recall = recall_score(y_val, y_pred, average="weighted")
        F1_score = f1_score(y_val, y_pred, average="weighted")

        # For ROC AUC, use predict_proba if available
        try:
            y_prob = model.predict_proba(X_val)
            # One-vs-Rest for multiclass
            roc_auc = roc_auc_score(y_val, y_prob, multi_class="ovr")  
        except AttributeError:
            logging.warning("Model does not support predict_proba; skipping ROC AUC.")
            roc_auc = None

        # Confusion matrix
        Confusion_matrix = confusion_matrix(y_val, y_pred)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": F1_score,
            "ROC AUC": roc_auc,
            "True Negatives": Confusion_matrix[0][0],
            "False Positives": Confusion_matrix[0][1],
            "False Negatives": Confusion_matrix[1][0],
            "True Positives": Confusion_matrix[1][1],

        }
        logging.info(f"Model Evaluation Metrics: {metrics}")
        return metrics["F1 Score"]
    
# Context Class for Model Evaluation
class ModelEvaluator:
    def __init__(self, strategy:ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy.

        Parameters:
            strategy (ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator.

        Parameters:
            strategy (ModelEvaluationStrategy): The new strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy.")
        self._strategy = strategy

    def evaluate(self, model: ClassifierMixin, X_val: pd.DataFrame, y_val: pd.Series) -> Any:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
            model (ClassifierMixin): The trained model to evaluate.
            X_test (pd.DataFrame): The testing data features.
            y_test (pd.Series): The testing data labels/target.

        Returns:
            Any: A dictionary containing evaluation metrics or a single evaluation metric.
        """
        logging.info("Evaluating the model using the selected strategy.")
        return self._strategy.evaluate_model(model, X_val, y_val)
    

# Example usage
if __name__ == "__main__":
    pass