"""
    Features explorer.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

from fet import flow


class Explorer:
    """Dataset explorer.

    Args:
        y (str, optional): Target/dependent variable. Defaults to None.
    """

    def __init__(self, y=None):
        self.y = y
        self.feature_cols = []

    def fit(self, df):
        """Fit DataFrame to Explorer.

        Args:
            df (pandas.DataFrame): DataFrame to explore.
        """
        self.df = df.copy()
        self.df.columns = self.df.columns.str.lower()

        flow.extract_per_flow_stats(self.df, inplace=True)
        self.feature_cols += flow.feature_cols

        self._remove_low_variance()

    def correlation_matrix(self):
        """Plot correlation matrix of feature columns."""
        _, ax = plt.subplots(figsize=(10, 8))

        corr = self.df[self.feature_cols].corr()
        sns.heatmap(corr, square=True, ax=ax)

        plt.show()

    def pairplot(self, cols):
        """Plot pairwise plot of feature columns (and target variable if present).

        Args:
            cols (list): List of columns to include.
        """
        if self.y:
            sns.pairplot(self.df[cols + [self.y]], hue=self.y)
        else:
            sns.pairplot(self.df[cols])

        plt.show()

    def _remove_low_variance(self, threshold=0):
        """Removes low variance features.

        Args:
            threshold (int, optional): Variance threshold. Defaults to 0.
        """
        sel = VarianceThreshold(threshold)
        sel.fit(self.df[self.feature_cols])

        self.feature_cols = list(pd.Index(self.feature_cols)[sel.get_support()])
