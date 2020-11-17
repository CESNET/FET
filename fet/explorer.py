"""
    Features explorer.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
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

    def histplot(self, cols=None, **kwargs):
        """Plot univariate histograms.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.histplot.
        """
        self._grid(sns.histplot, cols=cols, hue=self.y, **kwargs)

    def kdeplot(self, cols=None, **kwargs):
        """Plot univariate kernel density estimations.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.kdeplot.
        """
        self._grid(sns.kdeplot, cols=cols, hue=self.y, **kwargs)

    def ecdfplot(self, cols=None, **kwargs):
        """Plot empirical cumulative distribution functions.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.ecdfplot.
        """
        self._grid(sns.ecdfplot, cols=cols, hue=self.y, **kwargs)

    def stripplot(self, cols=None, **kwargs):
        """Plot scatter plots.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.stripplot.
        """
        self._grid(sns.stripplot, cols=cols, y=self.y, **kwargs)

    def boxplot(self, cols=None, **kwargs):
        """Plot box plots.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.boxplot.
        """
        self._grid(sns.boxplot, cols=cols, y=self.y, **kwargs)

    def violinplot(self, cols=None, **kwargs):
        """Plot violin plots - combination of boxplot and kde.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.violinplot.
        """
        self._grid(sns.violinplot, cols=cols, y=self.y, **kwargs)

    def boxenplot(self, cols=None, **kwargs):
        """Plot boxen plot - enhanced box plot.

        Args:
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
            **kwargs: Other keyword arguments for seaborn.boxenplot.
        """
        self._grid(sns.boxenplot, cols=cols, y=self.y, **kwargs)

    def _grid(self, func, cols=None, **kwargs):
        """Grid interface for univariate plotting.

        Args:
            func (callable): Callback plotting function.
            cols (list, optional): List of columns to include.
                Defaults to None - which includes all feature columns.
        """
        if not cols:
            cols = self.feature_cols

        if len(cols) < 4:
            ncols = len(cols)
        elif len(cols) % 3 == 0:
            ncols = 3
        elif len(cols) % 2 == 0:
            ncols = 2
        else:
            ncols = 3

        ny = len(self.df["y"].unique())

        nrows = math.ceil(len(cols) / ncols)
        vsize = nrows * 3 if ny == 1 else nrows * ny
        hsize = 4.5 * ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(hsize, vsize))
        fig.tight_layout(w_pad=2, h_pad=3)

        if len(cols) == 1:
            func(x=cols[0], data=self.df, ax=axes, **kwargs)
        else:
            i = 0
            for row_ax in axes:
                if i >= len(cols):
                    break

                if type(row_ax) == np.ndarray:
                    for col_ax in row_ax:
                        if i >= len(cols):
                            break

                        func(x=cols[i], data=self.df, ax=col_ax, **kwargs)
                        i += 1
                else:
                    func(x=cols[i], data=self.df, ax=row_ax, **kwargs)
                    i += 1

        plt.show()

    def _remove_low_variance(self, threshold=0):
        """Remove low variance features.

        Args:
            threshold (int, optional): Variance threshold. Defaults to 0.
        """
        sel = VarianceThreshold(threshold)
        sel.fit(self.df[self.feature_cols])

        self.feature_cols = list(pd.Index(self.feature_cols)[sel.get_support()])
