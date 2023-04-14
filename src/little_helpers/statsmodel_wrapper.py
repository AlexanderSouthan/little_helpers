# -*- coding: utf-8 -*-

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.base import BaseEstimator, RegressorMixin

class statsmodel_wrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors

    This is from https://stackoverflow.com/questions/41045752/using-statsmodel-estimations-with-scikit-learn-cross-validation-is-it-possible
    """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)

class ols_wrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_string, response_name):
        self.model_string = model_string
        self.response_name = response_name
    def fit(self, X, y):
        X[self.response_name] = y
        self.model_ = ols(self.model_string, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        return self.results_.predict(X)
