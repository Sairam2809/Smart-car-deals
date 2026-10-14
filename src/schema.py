"""Lightweight input metadata: safe to import without loading ML libraries."""
TARGET = 'transmission'
REFERENCE_YEAR = 2026
RAW_FEATURES = ['brand','model','year','price_gbp','mileage_miles','fuel_type','tax_gbp','mpg_uk','engine_size_l']
NUMERIC_INPUTS = ['year','price_gbp','mileage_miles','tax_gbp','mpg_uk','engine_size_l']
NUMERIC_FEATURES = ['car_age','log_price','log_mileage','tax_gbp','mpg_uk','engine_size_l']
CATEGORICAL_FEATURES = ['brand','model','fuel_type']
PART_B = {'random_forest':'Random Forest','adaboost':'AdaBoost','gradient_boosting':'Gradient Boosting','bagging':'Bagging','mlp':'MLP'}
