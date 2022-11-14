import pickle

import numpy as np

from .transformers import (
    bins_features_fit,
    bins_features_transform,
    cyclical_features_fit,
    cyclical_features_transform,
    math_features_fit,
    math_features_transform,
)


class FeatureCreation:
    def __init__(self):
        self.transformations = {}

    def fit(self, X, y, clf, categorical_columns=[], numerial_columns=[], verbose=1):
        categorical_columns = (
            categorical_columns
            if len(categorical_columns)
            else self._get_categorical_columns(X)
        )
        numerial_columns = (
            numerial_columns
            if len(numerial_columns)
            else self._get_numerical_columns(X)
        )

        trans_df = X.copy()

        trans_df, self.transformations["bins"] = bins_features_fit(
            trans_df, y, numerial_columns, clf, verbose
        )
        trans_df, self.transformations["cyclical"] = cyclical_features_fit(
            trans_df, y, numerial_columns, clf, verbose
        )
        trans_df, self.transformations["math"] = math_features_fit(
            trans_df, y, numerial_columns, clf, verbose
        )

        return trans_df

    def transform(self, df):
        trans_df = df.copy()

        bins_features_transform(trans_df, self.transformations["bins"])
        cyclical_features_transform(trans_df, self.transformations["cyclical"])
        math_features_transform(trans_df, self.transformations["math"])

        return trans_df

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.transformations, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.transformations = pickle.load(f)

    def _get_categorical_columns(self, df):
        return list(df.select_dtypes(include=["object"]).columns)

    def _get_numerical_columns(self, df):
        return list(df.select_dtypes(include=[np.number]).columns)
