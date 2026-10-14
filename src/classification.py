"""Shared, fold-safe classification preparation, evaluation and figures."""
from pathlib import Path
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from src.schema import TARGET, RAW_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.features import CarFeatures

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
REFERENCE_YEAR = 2026
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {'f1_macro': 'f1_macro', 'f1_weighted': 'f1_weighted',
           'balanced_accuracy': 'balanced_accuracy'}

def load_data():
    raw = pd.read_csv(ROOT / 'data/classification_cars.csv')
    df = raw.copy()
    df[TARGET] = df[TARGET].astype('string').str.strip()
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    if set(df[TARGET]) != {'Automatic', 'Manual'}:
        raise ValueError('Expected the two transmission labels: Automatic and Manual.')
    train, test = train_test_split(df, test_size=.2, stratify=df[TARGET], random_state=SEED)
    assert set(train.index).isdisjoint(test.index)
    return raw, df, train, test

def build_pipeline(estimator):
    numeric = NUMERIC_FEATURES
    categorical = CATEGORICAL_FEATURES
    prep = ColumnTransformer([
        ('num', SkPipeline([('imputer', SimpleImputer(strategy='median')),
                            ('scale', StandardScaler())]), numeric),
        ('cat', SkPipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ])
    return Pipeline([('features', CarFeatures()), ('preprocess', prep),
                     ('model', estimator)])

def configurations():
    # Baseline candidates are included in every search. A test-score increase is not guaranteed.
    return {
        'random_forest': ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1), [
            {'model__n_estimators':[100], 'model__max_depth':[None], 'model__min_samples_leaf':[1]},
            {'model__n_estimators':[150,250], 'model__max_depth':[8,None], 'model__min_samples_leaf':[2]}]),
        'adaboost': ('AdaBoost', AdaBoostClassifier(random_state=SEED), [
            {'model__n_estimators':[50], 'model__learning_rate':[1.]},
            {'model__n_estimators':[100,200], 'model__learning_rate':[.1,.5]}]),
        'gradient_boosting': ('Gradient Boosting', GradientBoostingClassifier(random_state=SEED), [
            {'model__n_estimators':[100], 'model__learning_rate':[.1], 'model__max_depth':[3]},
            {'model__n_estimators':[100,200], 'model__learning_rate':[.05,.1], 'model__max_depth':[2]}]),
        'bagging': ('Bagging', BaggingClassifier(estimator=DecisionTreeClassifier(random_state=SEED),
                            n_estimators=10, random_state=SEED, n_jobs=1), [
            {'model__n_estimators':[10], 'model__max_samples':[1.], 'model__estimator__max_depth':[None]},
            {'model__n_estimators':[50,100], 'model__max_samples':[.8], 'model__estimator__max_depth':[8,None]}]),
        'mlp': ('MLP', MLPClassifier(hidden_layer_sizes=(100,), max_iter=800, early_stopping=False,
                      tol=.001, n_iter_no_change=25, random_state=SEED), [
            {'model__hidden_layer_sizes':[(100,)], 'model__activation':['relu'], 'model__alpha':[.0001]},
            {'model__hidden_layer_sizes':[(50,),(64,32)], 'model__activation':['relu','tanh'], 'model__alpha':[.01]}])
    }

def part_a_models():
    return {
        'logistic_regression': ('Logistic Regression', LogisticRegression(max_iter=3000, random_state=SEED)),
        'knn': ('KNN', KNeighborsClassifier()),
        'naive_bayes': ('Gaussian Naive Bayes', GaussianNB()),
        'decision_tree': ('Decision Tree', DecisionTreeClassifier(random_state=SEED)),
        'svc': ('SVC', SVC(probability=True, random_state=SEED))
    }

def automatic_probability(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index('Automatic')]

def metrics(model, X, y):
    pred = model.predict(X)
    binary = (y == 'Automatic').astype(int)
    p = automatic_probability(model, X)
    return {
        'Accuracy': accuracy_score(y,pred),
        'Precision_weighted': precision_score(y,pred,average='weighted',zero_division=0),
        'Recall_weighted': recall_score(y,pred,average='weighted',zero_division=0),
        'F1_weighted': f1_score(y,pred,average='weighted',zero_division=0),
        'F1_macro': f1_score(y,pred,average='macro',zero_division=0),
        'Balanced_accuracy': balanced_accuracy_score(y,pred),
        'Automatic_precision': precision_score(y,pred,pos_label='Automatic',zero_division=0),
        'Automatic_recall': recall_score(y,pred,pos_label='Automatic',zero_division=0),
        'ROC_AUC': roc_auc_score(binary,p),
        'Average_precision_Automatic': average_precision_score(binary,p)
    }

def evaluate_pair(slug, name, baseline, tuned, X_test, y_test, baseline_cv, search, warning_messages):
    out = ROOT/'results'/slug
    out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for stage, model, mean, sd in [
        ('Before tuning',baseline, np.mean(baseline_cv['test_f1_macro']),np.std(baseline_cv['test_f1_macro'])),
        ('After tuning',tuned,search.best_score_,search.cv_results_['std_test_f1_macro'][search.best_index_])]:
        rows.append({'Model':name,'Stage':stage, **metrics(model,X_test,y_test),
                     'CV_macro_F1':float(mean),'CV_macro_F1_std':float(sd)})
        joblib.dump(model,ROOT/'models'/f'{slug}_{"before" if stage=="Before tuning" else "after"}.joblib',compress=3)
        report=pd.DataFrame(classification_report(y_test,model.predict(X_test),output_dict=True,zero_division=0)).T
        report.to_csv(out/f'{"before" if stage=="Before tuning" else "after"}_class_report.csv')
    table=pd.DataFrame(rows)
    table.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(search.cv_results_).to_csv(out/'search_results.csv',index=False)
    pd.DataFrame({k:v for k,v in baseline_cv.items() if k.startswith('test_')}).to_csv(out/'baseline_cv_folds.csv',index=False)
    metadata={'best_parameters':search.best_params_, 'selection_metric':'5-fold training CV macro F1',
              'positive_class':'Automatic', 'warnings':sorted(set(warning_messages)),
              'test_used_for_tuning':False}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str))
    return table

def plot_pair(slug, baseline, tuned, X, y):
    out=ROOT/'results'/slug
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,model,title in zip(axes,[baseline,tuned],['Before tuning','After tuning']):
        ConfusionMatrixDisplay.from_estimator(model,X,y,display_labels=['Automatic','Manual'],
            labels=['Automatic','Manual'],ax=ax,cmap='Blues',colorbar=False)
        ax.set_title(title)
    fig.suptitle('Held-out test confusion matrices');fig.tight_layout()
    fig.savefig(out/'confusion_matrices.png',dpi=130);plt.show()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for model,title in [(baseline,'Before tuning'),(tuned,'After tuning')]:
        p=automatic_probability(model,X);b=(y=='Automatic').astype(int)
        RocCurveDisplay.from_predictions(b,p,name=title,ax=axes[0])
        PrecisionRecallDisplay.from_predictions(b,p,name=title,ax=axes[1])
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Chance');axes[0].legend()
    axes[1].axhline((y=='Automatic').mean(),ls='--',color='gray',label='Automatic prevalence');axes[1].legend()
    axes[0].set_title('ROC — Automatic is positive');axes[1].set_title('Precision–recall — Automatic')
    fig.tight_layout();fig.savefig(out/'roc_pr.png',dpi=130);plt.show()
    est=tuned.named_steps['model']
    if hasattr(est,'feature_importances_'):
        names=tuned.named_steps['preprocess'].get_feature_names_out()
        s=pd.Series(est.feature_importances_,index=names).nlargest(15).sort_values()
        fig,ax=plt.subplots(figsize=(8,5));s.plot.barh(ax=ax,color='#128276')
        ax.set(xlabel='Impurity-based importance',title='Tuned model feature importance')
        fig.tight_layout();fig.savefig(out/'feature_importance.png',dpi=130);plt.show()
    if hasattr(est,'loss_curve_'):
        fig,ax=plt.subplots(figsize=(7,3));ax.plot(est.loss_curve_)
        ax.set(xlabel='Iteration',ylabel='Training loss',title='MLP optimization history')
        fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=130);plt.show()

def audit_plots(df, train):
    print('Cleaned shape:',df.shape, '| Training:',len(train),'| Test:',len(df)-len(train))
    print('Missing training values:\n',train.isna().sum().to_string())
    fig,ax=plt.subplots(figsize=(6,3))
    train.transmission.value_counts().reindex(['Automatic','Manual']).plot.bar(ax=ax,color=['#c7353f','#435f7d'])
    ax.set(title='Training transmission counts — no oversampling',xlabel='Transmission',ylabel='Listings')
    ax.tick_params(axis='x',rotation=0);fig.tight_layout();plt.show()
    numeric=CarFeatures().transform(train[RAW_FEATURES]).select_dtypes(include='number')
    numeric.hist(figsize=(12,6),bins=25,color='#c7353f')
    plt.suptitle('Training feature distributions');plt.tight_layout();plt.show()
    fig,ax=plt.subplots(figsize=(7,5));sns.heatmap(numeric.corr(),annot=True,fmt='.2f',cmap='coolwarm',center=0,ax=ax)
    ax.set_title('Training numeric correlations');fig.tight_layout();plt.show()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,x in zip(axes,['year','mileage_miles']):
        sns.scatterplot(data=train,x=x,y='price_gbp',hue='transmission',alpha=.5,ax=ax)
        ax.set_title(f'Price (GBP) and {x} by transmission')
    fig.tight_layout();plt.show()
    q1=numeric.quantile(.25);q3=numeric.quantile(.75);iqr=q3-q1
    print('Training IQR outlier counts (retained; log transforms reduce skew):')
    print(((numeric<q1-1.5*iqr)|(numeric>q3+1.5*iqr)).sum().to_string())
