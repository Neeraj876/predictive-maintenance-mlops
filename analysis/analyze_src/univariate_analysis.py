from abc import ABC, abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the abstract base class or a common interface for univariate analysis strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass

# Implement a concrete class/strategy to analyze numerical features by plotting their distribution.

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def Analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with a KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
        
# Implement a concrete class/strategy to analyze categorical features by plotting their frequency distribution.

class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categorical feature using a count plot

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a count plot showing the frequency of each category.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette='muted')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()
        
# Implement the context class/strategy that uses UnivariateAnalysisStrategy
# -------------------------------------------------
# This class allows to switch between different univariate analysis strategy

class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the current strategy's analysis method and visualizes the results
        """
        self._strategy.analyze(df, feature)


# Example Usage
if __name__ == "__main__":
    # # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize the UnivariateAnalyzer with a specific strategy
    
    # Analyzing a numerical feature
    # analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'SalePrice')

    # Analyzing a categorical feature
    # analyzer.set_strategy(CategoricalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Neighborhood')
    pass
