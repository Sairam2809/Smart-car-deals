"""Input checks shared by batch predictions and tests."""
import numpy as np
import pandas as pd
from src.schema import RAW_FEATURES,NUMERIC_INPUTS

def validate_batch(df):
    missing=set(RAW_FEATURES)-set(df.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if df.empty:raise ValueError('The CSV has no rows.')
    if len(df)>10000:raise ValueError('Upload at most 10,000 rows at once.')
    if df.columns.duplicated().any():raise ValueError('Column names must be unique.')
    df=df.copy()
    for col in NUMERIC_INPUTS:
        original=df[col];df[col]=pd.to_numeric(df[col],errors='coerce')
        if ((original.notna()&df[col].isna())|np.isinf(df[col])).any():
            raise ValueError(f'{col}: use plain finite numbers without symbols or commas.')
    if (df.year.notna()&(~df.year.between(1970,2026)|(df.year%1!=0))).any():
        raise ValueError('year must be a whole number between 1970 and 2026.')
    if (df[['price_gbp','mpg_uk','engine_size_l']]<=0).any().any():
        raise ValueError('Price, MPG and engine size must be positive.')
    if (df[['mileage_miles','tax_gbp']]<0).any().any():
        raise ValueError('Mileage and tax cannot be negative.')
    return df

def validate_regression_batch(df):
    from src.schema import REGRESSION_RAW_FEATURES, REGRESSION_NUMERIC_FEATURES
    missing = set(REGRESSION_RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError('Missing regression columns: ' + ', '.join(sorted(missing)))
    if df.empty:
        raise ValueError('The CSV has no rows.')
    if len(df) > 10000:
        raise ValueError('Upload at most 10,000 rows at once.')
    if df.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    df = df.copy()
    for col in REGRESSION_NUMERIC_FEATURES:
        original = df[col]
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if ((original.notna() & df[col].isna()) | np.isinf(df[col])).any():
            raise ValueError(f'{col}: use plain finite numbers without symbols or commas.')
    if (df[['km_driven', 'mileage', 'engine', 'max_power']] < 0).any().any():
        raise ValueError('KM driven, mileage, engine, and max power cannot be negative.')
    return df
