from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureProjection(BaseEstimator, TransformerMixin):
    """
    Get a list of fields and project them as lists or dictionaries
    """

    def __init__(self, fields: list[str], as_dict=False, convert_na=True):
        self.fields = fields
        self.as_dict = as_dict
        self.convert_na = convert_na

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        res = []
        if self.as_dict:
            return X[self.fields].to_dict(orient='records')
        else:
            return X[self.fields].values.tolist()


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Map a categorical field to a 4-dimensional vector
    [mean(y), std(y), percentile(y, 5), percentile(y, 95)]
    Use a default value if the category is not present in the training set or if the frequency is too low
    """

    def __init__(self, categorical_field, min_freq=5):
        self.min_freq = min_freq
        self.categorical_field = categorical_field
        self.stats_ = None
        self.default_stats = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        values = defaultdict(list)
        for val in X[self.categorical_field].unique():
            # Get the target values for each category
            values[val] = y[ X[self.categorical_field] == val ].values 

        self.stats_ = {}
        for cat_value, tar_values in values.items():
            if len(tar_values) < self.min_freq: continue
            tar_values = np.asarray(tar_values)
            self.stats_[cat_value] = [
                np.mean(tar_values), np.std(tar_values),
                np.percentile(tar_values, 90), np.percentile(tar_values, 10),
            ]

        self.default_stats_ = [
            np.mean(y), np.std(y),
            np.percentile(y, 90), np.percentile(y, 10)
        ]
        return self

    def transform(self, X: pd.DataFrame):
        res = []
        return X[self.categorical_field].apply(lambda x: self.stats_.get(x, self.default_stats_)).values.tolist()
        for i, doc in enumerate(X):
            vector = self.stats_.get(doc[self.categorical_field], self.default_stats_)
            res.append(vector)
        return res
