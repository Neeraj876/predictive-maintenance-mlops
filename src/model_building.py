import logging
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import joblib
from config import RANDOM_STATE
import re

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Model Building Strategy
class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Abstract method to build and train a model.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Any: A trained classification model instance.
        """
        pass

# Concrete Strategy for Logistic Regression
class LogisticRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a Logistic Regression model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            LogisticRegression: A trained Logistic Regression model.
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Training the Logistic Regression Model.")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/logistic.pkl")
        logging.info("Logistic Regression training completed.")
        return model
        
# Concrete Strategy for SVM
class SVCStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a Support Vector Classifier model on the provided training data, optionally with fine-tuning.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            SVC: A trained SVC model (either fine-tuned or default).
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")
        
        logging.info("Started training the SVM model.")
        model = SVC(C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/svm.pkl")
        logging.info("Completed training the SVM model.")
        return model
            
# Concrete Strategy for Naive Bayes
class NaiveBayesStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a Naive Bayes model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            MultinomialNB: A trained Naive Bayes model.
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")
        
        logging.info("Training the Naive Bayes model.")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/NBayes.pkl")
        logging.info("Completed training the Naive Bayes model.")
        return model
    
# Concrete Strategy for Random Forest
class RandomForestStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Trains a Random Forest model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            RandomForestClassifier: A trained Random Forest model.
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas Series.")

        logging.info("Training the Random Forest model.")
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/rf.pkl")
        logging.info("Completed training the Random Forest model.")
        return model
    
# Concrete Strategy for Binary Classification using XGBOOST
# -------------------------------------------
class BinaryModelTrainingStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train a XGBClassifier model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            XGBClassifier: A trained xgboost binary classification model.
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train_binary must be a pandas Series.")
        
        logging.info("Initializing Binary classification model.")

        logging.info("Cleaning column names in X_train.")

        # Clean column names in X_train to avoid special characters issues
        X_train.columns = [re.sub(r"[<>[\]]", "", col) for col in X_train.columns]
        print("X_train_binary Cleaned column names:", X_train.columns)

        # Binary Classification Model
        model = XGBClassifier(random_state=RANDOM_STATE)

        logging.info("Training Binary Classification model.")

        # Train the model
        print("X_train_binary before fitting to the model", X_train)
        print('Expected X_train_binary columns', X_train.columns)
        model.fit(X_train, y_train) 

        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/xgb.pkl")

        #logging.info("Binary classification model training completed.")
        return model
    
# Concrete Strategy for Multiclass Classification
# -----------------------------------------------
class MulticlassModelTrainingStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train a XGBClassifier model on the provided training data.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            XGBClassifier: A trained xgboost multiclass classification model.
        """

        # Ensure the inputs are of the correct type
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train_binary must be a pandas Series.")

        logging.info("Initializing Multiclass classification model.")

        #multiclass_predictions = np.zeros(len(X))
        X_train.columns = [re.sub(r"[<>[\]]", "", col) for col in X_train.columns]
        print("X_train_multiclass_resampled Cleaned column names:", X_train.columns)

        # Multiclass Classification Model
        model = XGBClassifier(random_state=RANDOM_STATE)

        logging.info("Training Multiclass classification model.")

        # Train the model
        print("X_train_multiclass_resampled before fitting to the model", X_train)
        print('Expected X_train_multiclass_resampled columns', X_train.columns)
        model.fit(X_train, y_train)

        joblib.dump(model, "/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/saved_models/multi.pkl")

        #logging.info("Multi class model training completed.")
        return model
    
# Context Class for Model Building Strategy
class ModelBuilder:
    def __init__(self, strategy:ModelBuildingStrategy):
        """
        Initializes the ModelBuildingStrategy with the X_train, y_train, fine_tuning and a strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.
        """
        self._strategy = strategy

    def set_strategy(self, strategy:ModelBuildingStrategy):
        """
        Set the model building strategy.

        Parameters:
            strategy (ModelBuildingStrategy): The strategy to set.
        """
        self._strategy = strategy

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model using the current strategy.

        Parameters:
            X_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data labels/target.

        Returns:
            Any: A trained model instance from the chosen strategy.
        """
        return self._strategy.build_and_train_model(X_train, y_train)

# Example usage
if __name__ == "__main__":
    pass




    
