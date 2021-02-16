"""
    Features explorer.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

from fet import flow


class Explorer:
    """Dataset explorer.

    Args:
        y (str, optional): Target/dependent variable. Defaults to None.
    """

    def __init__(self, y=None):
        if y:
            self.y = y.lower()
        else:
            self.y = y

        self.feature_cols = []

    def fit(self, df, flow_features=True):
        """Fit DataFrame to Explorer.

        Args:
            df (pandas.DataFrame): DataFrame to explore.
        """
        self.df = df.copy()

        if "ipaddr DST_IP" in self.df.columns:
            self.df.rename(columns=lambda x: x.split()[1], inplace=True)

        self.df.columns = self.df.columns.str.lower()

        if flow_features:
            flow.extract_per_flow_stats(self.df, inplace=True)
            self.feature_cols += flow.feature_cols

        self._remove_low_variance()

    def remove_features(self, cols):
        """Remove features from feature vector.

        Args:
            cols (list): List of column names to remove.
        """
        self.feature_cols = [x for x in self.feature_cols if x not in cols]

    def kbest(self, k, score_func=None):
        """Select k highest features according to scoring function.

        Args:
            k (int): Number of top features to select.
            score_func ([callable], optional): Scoring function. Defaults
                to None - which uses f_classif.

        Raises:
            ValueError: Nothing to classify w/o target variable.

        Returns:
            list: Unsorted list of k best features.
        """
        if not self.y:
            raise ValueError("No target variable.")

        if not score_func:
            score_func = f_classif

        sel = SelectKBest(score_func, k=k)
        sel.fit(self.df[self.feature_cols], self.df[self.y])

        return list(pd.Index(self.feature_cols)[sel.get_support()])

    def feature_scores(self, score_func=None):
        """Evaluate feature scores using scoring function.

        Args:
            score_func ([callable], optional): Scoring function. Defaults
                to None - which uses f_classif.

        Raises:
            ValueError: Nothing to classify w/o target variable.

        Returns:
            list: Sorted list of tuples (feature, score).
        """
        if not self.y:
            raise ValueError("No target variable.")

        if not score_func:
            score_func = f_classif

        scores, _ = score_func(self.df[self.feature_cols], self.df[self.y])

        res = []

        for i, score in enumerate(scores):
            res.append((self.feature_cols[i], score))

        res = sorted(res, key=lambda x: x[1], reverse=True)

        return res

    def feature_importances(self, clf=None):
        """Evaluate feature importances using classifier.

        Args:
            clf (object, optional): Instantiated classifier. Defaults
                to None - which uses ExtraTreesClassifier.

        Raises:
            ValueError: Nothing to classify w/o target variable.

        Returns:
            list: Sorted list of tuples (feature, importance).
        """
        if not self.y:
            raise ValueError("No target variable.")

        if clf is None:
            clf = ExtraTreesClassifier(n_estimators=50)

        clf = clf.fit(self.df[self.feature_cols], self.df[self.y])

        res = []

        for i, importance in enumerate(clf.feature_importances_):
            res.append((self.feature_cols[i], importance))

        res = sorted(res, key=lambda x: x[1], reverse=True)

        return res

    def plot_feature_scores(self, clf=None):
        """Plot feature scores using scoring function.

        Args:
            clf (object, optional): Instantiated classifier. Defaults
                to None - which uses ExtraTreesClassifier.
        """
        scores = self.feature_scores(clf)
        n = len(scores)

        plt.subplots(figsize=(16, 5))
        plt.bar(range(n), [v for _, v in scores])
        plt.xticks(range(n), [k for k, _ in scores], rotation=90)

        plt.show()

    def plot_feature_importances(self, clf=None):
        """Plot feature importances using classifier.

        Args:
            clf (object, optional): Instantiated classifier. Defaults
                to None - which uses ExtraTreesClassifier.
        """
        importances = self.feature_importances(clf)
        n = len(importances)

        plt.subplots(figsize=(16, 5))
        plt.bar(range(n), [v for _, v in importances])
        plt.xticks(range(n), [k for k, _ in importances], rotation=90)

        plt.show()

    def plot_pca(self):
        """Plot scatterplot of the first two principal components."""
        pca = PCA(n_components=2)
        data = pca.fit_transform(self.df[self.feature_cols])
        projection = pd.DataFrame(data=data, columns=("x", "y"))

        if self.y:
            projection[self.y] = self.df[self.y]
            sns.relplot(x="x", y="y", hue=self.y, data=projection)
        else:
            sns.relplot(x="x", y="y", data=projection)

        plt.show()

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

        ny = len(self.df[self.y].unique())

        nrows = math.ceil(len(cols) / ncols)
        vsize = nrows * 3 if ny == 1 else nrows * ny
        hsize = 4.5 * ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(hsize, vsize))
        fig.tight_layout(w_pad=5, h_pad=3)

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
