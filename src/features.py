"""Deterministic UK-car feature transform stored inside fitted pipelines."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.schema import NUMERIC_INPUTS,NUMERIC_FEATURES,CATEGORICAL_FEATURES,REFERENCE_YEAR

class CarFeatures(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):return self
    def transform(self,X):
        X=X.copy()
        for col in NUMERIC_INPUTS:X[col]=pd.to_numeric(X[col],errors='coerce')
        X['car_age']=REFERENCE_YEAR-X.year
        X['log_price']=np.log1p(X.price_gbp.clip(lower=0))
        X['log_mileage']=np.log1p(X.mileage_miles.clip(lower=0))
        for col in CATEGORICAL_FEATURES:
            X[col]=X[col].astype('string').str.strip().replace('',pd.NA).astype(object)
        return X[NUMERIC_FEATURES+CATEGORICAL_FEATURES].replace({pd.NA:np.nan})
