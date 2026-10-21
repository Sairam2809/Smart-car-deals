"""Lightweight input metadata: safe to import without loading ML libraries."""
TARGET = 'transmission'
REFERENCE_YEAR = 2026
RAW_FEATURES = ['brand','model','year','price_gbp','mileage_miles','fuel_type','tax_gbp','mpg_uk','engine_size_l']
NUMERIC_INPUTS = ['year','price_gbp','mileage_miles','tax_gbp','mpg_uk','engine_size_l']
NUMERIC_FEATURES = ['car_age','log_price','log_mileage','tax_gbp','mpg_uk','engine_size_l']
CATEGORICAL_FEATURES = ['brand','model','fuel_type']
PART_B = {'random_forest':'Random Forest','adaboost':'AdaBoost','gradient_boosting':'Gradient Boosting','bagging':'Bagging','mlp':'MLP'}

# Review 1 Track: 10 Regression Models for Vehicle Price Prediction
REGRESSION_TARGET = 'selling_price'
REGRESSION_NUMERIC_FEATURES = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
REGRESSION_CATEGORICAL_FEATURES = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
REGRESSION_RAW_FEATURES = REGRESSION_NUMERIC_FEATURES + REGRESSION_CATEGORICAL_FEATURES
REGRESSION_MODELS = {
    'random_forest_regressor': 'Random Forest Regressor',
    'gradient_boosting_regressor': 'Gradient Boosting Regressor',
    'knn_regressor': 'K-Nearest Neighbors Regressor',
    'decision_tree_regressor': 'Decision Tree Regressor',
    'linear_regression': 'Linear Regression',
    'ridge_regression': 'Ridge Regression',
    'lasso_regression': 'Lasso Regression',
    'elasticnet_regression': 'ElasticNet Regression',
    'polynomial_regression': 'Polynomial Regression (Degree 2)',
    'svr': 'Support Vector Regressor (SVR)'
}
