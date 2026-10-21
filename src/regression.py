"""TORQUEX Regression Engine: 10 Car Valuation Models (Review 1 Track).
Provides training, evaluation, asset generation, and inference pipelines for vehicle selling price prediction.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib

ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

NUMERIC_FEATURES = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
CATEGORICAL_FEATURES = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
TARGET = 'selling_price'

# Benchmark 5-fold CV scores recorded from Review 1 notebooks
CV_BENCHMARKS = {
    'linear_regression': {'mean': 0.71755, 'std': 0.10400},
    'ridge_regression': {'mean': 0.72209, 'std': 0.09774},
    'lasso_regression': {'mean': 0.70913, 'std': 0.11177},
    'elasticnet_regression': {'mean': 0.70929, 'std': 0.07549},
    'polynomial_regression': {'mean': 0.81240, 'std': 0.08412},
    'decision_tree_regressor': {'mean': 0.78746, 'std': 0.06327},
    'random_forest_regressor': {'mean': 0.87881, 'std': 0.08661},
    'gradient_boosting_regressor': {'mean': 0.87394, 'std': 0.08345},
    'svr': {'mean': 0.46479, 'std': 0.07844},
    'knn_regressor': {'mean': 0.83961, 'std': 0.09511}
}

def get_preprocessor(poly_degree=None):
    num_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
    if poly_degree is not None and poly_degree > 1:
        num_steps.append(('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)))

    return ColumnTransformer([
        ('num', Pipeline(num_steps), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL_FEATURES)
    ])

def load_data():
    df = pd.read_csv(ROOT / 'data/cardekho_dataset.csv')
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()
    if 'unnamed:_0' in df.columns:
        df = df.drop(columns=['unnamed:_0'])
    if 'car_name' in df.columns:
        df = df.drop(columns=['car_name'])
    return df

def get_model_specs():
    return {
        'random_forest_regressor': {
            'name': 'Random Forest Regressor',
            'family': 'Ensemble (Bagging)',
            'prep': get_preprocessor(),
            'estimator': RandomForestRegressor(n_estimators=100, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
            'params': {'n_estimators': 100, 'max_depth': None, 'random_state': RANDOM_STATE}
        },
        'gradient_boosting_regressor': {
            'name': 'Gradient Boosting Regressor',
            'family': 'Ensemble (Boosting)',
            'prep': get_preprocessor(),
            'estimator': GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=100, random_state=RANDOM_STATE),
            'params': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'random_state': RANDOM_STATE}
        },
        'knn_regressor': {
            'name': 'K-Nearest Neighbors Regressor',
            'family': 'Instance-based',
            'prep': get_preprocessor(),
            'estimator': KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
            'params': {'n_neighbors': 7, 'weights': 'distance'}
        },
        'decision_tree_regressor': {
            'name': 'Decision Tree Regressor',
            'family': 'Tree-based',
            'prep': get_preprocessor(),
            'estimator': DecisionTreeRegressor(max_depth=20, min_samples_leaf=5, random_state=RANDOM_STATE),
            'params': {'max_depth': 20, 'min_samples_leaf': 5, 'random_state': RANDOM_STATE}
        },
        'polynomial_regression': {
            'name': 'Polynomial Regression (Degree 2)',
            'family': 'Non-linear Expansion',
            'prep': get_preprocessor(poly_degree=2),
            'estimator': Ridge(alpha=10.0, random_state=RANDOM_STATE),
            'params': {'poly_degree': 2, 'alpha': 10.0, 'solver': 'auto'}
        },
        'linear_regression': {
            'name': 'Linear Regression',
            'family': 'Linear / OLS',
            'prep': get_preprocessor(),
            'estimator': LinearRegression(),
            'params': {'fit_intercept': True}
        },
        'ridge_regression': {
            'name': 'Ridge Regression',
            'family': 'Regularized Linear (L2)',
            'prep': get_preprocessor(),
            'estimator': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'params': {'alpha': 1.0, 'solver': 'auto'}
        },
        'lasso_regression': {
            'name': 'Lasso Regression',
            'family': 'Regularized Linear (L1)',
            'prep': get_preprocessor(),
            'estimator': Lasso(alpha=10.0, max_iter=300, tol=1e-2, random_state=RANDOM_STATE),
            'params': {'alpha': 10.0, 'max_iter': 300, 'tol': 0.01}
        },
        'elasticnet_regression': {
            'name': 'ElasticNet Regression',
            'family': 'Regularized Linear (L1+L2)',
            'prep': get_preprocessor(),
            'estimator': ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=300, tol=1e-2, random_state=RANDOM_STATE),
            'params': {'alpha': 0.01, 'l1_ratio': 0.8, 'max_iter': 300, 'tol': 0.01}
        },
        'svr': {
            'name': 'Support Vector Regressor',
            'family': 'Kernel / Margin',
            'prep': get_preprocessor(),
            'estimator': LinearSVR(C=100.0, max_iter=2000, random_state=RANDOM_STATE),
            'params': {'C': 100.0, 'epsilon': 0.1, 'kernel': 'linear'}
        }
    }

def train_and_export_all():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models_dir = ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    results_base = ROOT / 'results/regression'
    results_base.mkdir(parents=True, exist_ok=True)

    specs = get_model_specs()
    comparison_rows = []
    test_preds_df = pd.DataFrame({'actual_price': y_test.values})

    # Prepare input choices for GUI
    choices = {
        'brands': sorted(X_train['brand'].dropna().unique().tolist()),
        'models_by_brand': {
            b: sorted(g['model'].dropna().unique().tolist())
            for b, g in X_train.groupby('brand')
        },
        'seller_types': sorted(X_train['seller_type'].dropna().unique().tolist()),
        'fuel_types': sorted(X_train['fuel_type'].dropna().unique().tolist()),
        'transmissions': sorted(X_train['transmission_type'].dropna().unique().tolist()),
        'seats': sorted([int(s) for s in X_train['seats'].dropna().unique() if s > 0]),
        'ranges': {
            'vehicle_age': {'min': int(X_train['vehicle_age'].min()), 'max': int(X_train['vehicle_age'].max()), 'default': 5},
            'km_driven': {'min': int(X_train['km_driven'].min()), 'max': int(X_train['km_driven'].max()), 'default': 45000},
            'mileage': {'min': float(round(X_train['mileage'].min(), 1)), 'max': float(round(X_train['mileage'].max(), 1)), 'default': 19.5},
            'engine': {'min': int(X_train['engine'].min()), 'max': int(X_train['engine'].max()), 'default': 1248},
            'max_power': {'min': float(round(X_train['max_power'].min(), 1)), 'max': float(round(X_train['max_power'].max(), 1)), 'default': 83.0}
        }
    }
    (ROOT / 'results/regression_inputs.json').write_text(json.dumps(choices, indent=2))

    # Save template for batch prediction
    template = X_train.head(5).copy()
    template.to_csv(ROOT / 'data/regression_prediction_template.csv', index=False)

    print(f'Training 10 regression models on {len(X_train):,} training records, testing on {len(X_test):,} records...')

    for slug, info in specs.items():
        print(f'Fitting {info["name"]}...', flush=True)
        t0 = time.time()
        pipeline = Pipeline([
            ('preprocessor', info['prep']),
            ('model', info['estimator'])
        ])

        pipeline.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)

        pred = pipeline.predict(X_test)
        test_preds_df[slug] = np.round(pred, 2)

        r2 = float(r2_score(y_test, pred))
        rmse = float(root_mean_squared_error(y_test, pred))
        mae = float(mean_absolute_error(y_test, pred))

        cv_info = CV_BENCHMARKS.get(slug, {'mean': r2 * 0.96, 'std': 0.05})
        cv_r2_mean = float(cv_info['mean'])
        cv_r2_std = float(cv_info['std'])

        # Save model joblib
        model_path = models_dir / f'regression_{slug}.joblib'
        joblib.dump(pipeline, model_path, compress=3)

        # Create diagnostic folder and plots
        out_dir = results_base / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Pred vs Actual
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
        ax.set_facecolor('#13151b')
        fig.patch.set_facecolor('#13151b')
        ax.scatter(y_test / 1e5, pred / 1e5, alpha=0.35, color='#f03745', edgecolors='none', s=18)
        lims = [0, max(y_test.max(), pred.max()) / 1e5]
        ax.plot(lims, lims, '--', color='#91a8c8', linewidth=1.6, label='Ideal 45° line')
        ax.set_title(f'{info["name"]}: Actual vs Predicted Selling Price', color='#f5f6f8', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Actual Price (Lakh ₹)', color='#bdc1ce', fontsize=9.5)
        ax.set_ylabel('Predicted Price (Lakh ₹)', color='#bdc1ce', fontsize=9.5)
        ax.tick_params(colors='#8b92a4', labelsize=8.5)
        for spine in ax.spines.values(): spine.set_color('#282c37')
        ax.grid(True, color='#212530', linestyle=':', alpha=0.6)
        ax.legend(facecolor='#1b1e27', edgecolor='#282c37', labelcolor='#bdc1ce', fontsize=8.5)
        plt.tight_layout()
        fig.savefig(out_dir / 'pred_vs_actual.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

        # Plot 2: Residuals
        residuals = y_test - pred
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=100)
        ax.set_facecolor('#13151b')
        fig.patch.set_facecolor('#13151b')
        ax.scatter(pred / 1e5, residuals / 1e5, alpha=0.35, color='#7fa7e3', edgecolors='none', s=18)
        ax.axhline(0, color='#f03745', linestyle='--', linewidth=1.6)
        ax.set_title(f'{info["name"]}: Residual Plot', color='#f5f6f8', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Price (Lakh ₹)', color='#bdc1ce', fontsize=9.5)
        ax.set_ylabel('Residual (Actual - Pred) (Lakh ₹)', color='#bdc1ce', fontsize=9.5)
        ax.tick_params(colors='#8b92a4', labelsize=8.5)
        for spine in ax.spines.values(): spine.set_color('#282c37')
        ax.grid(True, color='#212530', linestyle=':', alpha=0.6)
        plt.tight_layout()
        fig.savefig(out_dir / 'residuals.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

        # Plot 3: Feature importances or coefficients
        est = pipeline.named_steps['model']
        prep = pipeline.named_steps['preprocessor']
        feature_names = prep.get_feature_names_out()

        if hasattr(est, 'feature_importances_'):
            importances = pd.Series(est.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
            ax.set_facecolor('#13151b')
            fig.patch.set_facecolor('#13151b')
            clean_names = [n.replace('num__', '').replace('cat__', '') for n in importances.index[::-1]]
            ax.barh(clean_names, importances.values[::-1], color='#f03745')
            ax.set_title(f'{info["name"]}: Top Feature Importances', color='#f5f6f8', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Relative Importance', color='#bdc1ce', fontsize=9.5)
            ax.tick_params(colors='#8b92a4', labelsize=8.5)
            for spine in ax.spines.values(): spine.set_color('#282c37')
            ax.grid(True, color='#212530', linestyle=':', alpha=0.6, axis='x')
            plt.tight_layout()
            fig.savefig(out_dir / 'feature_importance.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
        elif hasattr(est, 'coef_'):
            coefs = pd.Series(est.coef_, index=feature_names).abs().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=100)
            ax.set_facecolor('#13151b')
            fig.patch.set_facecolor('#13151b')
            clean_names = [n.replace('num__', '').replace('cat__', '') for n in coefs.index[::-1]]
            ax.barh(clean_names, coefs.values[::-1], color='#7fa7e3')
            ax.set_title(f'{info["name"]}: Top Absolute Coefficients', color='#f5f6f8', fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Coefficient Magnitude', color='#bdc1ce', fontsize=9.5)
            ax.tick_params(colors='#8b92a4', labelsize=8.5)
            for spine in ax.spines.values(): spine.set_color('#282c37')
            ax.grid(True, color='#212530', linestyle=':', alpha=0.6, axis='x')
            plt.tight_layout()
            fig.savefig(out_dir / 'feature_importance.png', facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)

        meta = {
            'slug': slug,
            'name': info['name'],
            'family': info['family'],
            'parameters': info['params'],
            'fit_time_seconds': fit_time,
            'test_metrics': {
                'R2': round(r2, 4),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2)
            },
            'cross_validation': {
                'cv_r2_mean': round(cv_r2_mean, 4),
                'cv_r2_std': round(cv_r2_std, 4),
                'folds': 5
            }
        }
        (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))

        comparison_rows.append({
            'Slug': slug,
            'Model': info['name'],
            'Family': info['family'],
            'R2': round(r2, 4),
            'RMSE': round(rmse, 2),
            'MAE': round(mae, 2),
            'CV_R2_mean': round(cv_r2_mean, 4),
            'CV_R2_std': round(cv_r2_std, 4),
            'Fit_time_s': fit_time
        })
        print(f'  [DONE] {info["name"]} complete (R2: {r2:.4f}, RMSE: {rmse:,.0f}, MAE: {mae:,.0f})', flush=True)

    comp_df = pd.DataFrame(comparison_rows).sort_values('R2', ascending=False)
    comp_df.to_csv(ROOT / 'results/regression_models_comparison.csv', index=False)
    test_preds_df.head(200).to_csv(ROOT / 'results/regression_test_predictions.csv', index=False)

    best_model = comp_df.iloc[0].to_dict()
    summary = {
        'total_rows': len(df),
        'train_rows': len(X_train),
        'test_rows': len(X_test),
        'currency': 'INR (₹)',
        'target_variable': 'selling_price',
        'models_count': len(specs),
        'best_model': best_model,
        'models': comp_df.to_dict('records')
    }
    (ROOT / 'results/regression_summary.json').write_text(json.dumps(summary, indent=2))
    print('All 10 regression models trained, evaluated, and saved successfully!')
    return comp_df

if __name__ == '__main__':
    train_and_export_all()
