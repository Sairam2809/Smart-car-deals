"""Regenerate source notebooks. Run execute_notebooks.py to populate outputs."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(s)
def code(s):return nbf.v4.new_code_cell(s)
def save(name,cells):
    n=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.write(n,ROOT/'notebooks'/name)

BOOT='''from pathlib import Path
import sys, os, warnings, json
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent
if not (ROOT / 'src').exists():
    raise RuntimeError('Open this notebook inside the extracted project/notebooks folder.')
sys.path.insert(0, str(ROOT))
from threadpoolctl import threadpool_limits
thread_limit = threadpool_limits(limits=1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
%matplotlib inline
sns.set_theme(style='whitegrid', palette='colorblind')
'''
CLASS_IMPORT='''from sklearn.base import clone
from sklearn.model_selection import cross_validate, GridSearchCV
from src.classification import *
raw, df, train, test = load_data()
X_train, y_train = train[RAW_FEATURES], train[TARGET]
X_test, y_test = test[RAW_FEATURES], test[TARGET]
display(pd.DataFrame({'Raw': raw[TARGET].value_counts(), 'Cleaned': df[TARGET].value_counts(),
                      'Train': y_train.value_counts(), 'Test': y_test.value_counts()}))
display(df.dtypes.to_frame('dtype').astype(str))
print('Removed exact duplicate rows:', len(raw) - len(df))
assert not df.duplicated().any()
assert set(train.index).isdisjoint(test.index)
'''

names=[('random_forest','Random Forest'),('adaboost','AdaBoost'),('gradient_boosting','Gradient Boosting'),('bagging','Bagging'),('mlp','MLP Neural Network')]
for i,(slug,title) in enumerate(names,6):
    cells=[md(f'''# {title}: baseline and tuned classifier
**Review 2 · Classification Part B · target: transmission**

Predict Automatic vs Manual using 2,500 distinct real UK listing records (1,250 per class).
Prices are GBP and distance is miles. Source records come from Aditya Desai's UK used-car dataset;
no synthetic or repeated training rows are added. Distinct rows do not establish VIN-level uniqueness.

The fixed 80:20 stratified split gives 2,000 training rows and 500 test rows, with equal classes in both.
**No oversampling or class weights are needed.** Fit imputation, scaling and encoding inside each fold.
All Part A and B reruns share this new dataset and split; these are not the original Review 1 scores.
The balanced test sample measures performance for this sampled population, not natural market prevalence.

Selection uses **five-fold macro F1**, which gives both classes equal importance. Required weighted
metrics are also reported. The test set is used for the final before/after comparison, never to choose parameters.
The same feature engineering, preprocessing are used before and after tuning.
'''),code(BOOT+CLASS_IMPORT),md('''## Training-data audit and EDA
Missing numeric/categorical values, if present, are imputed within each training fold.
Car age uses a fixed reference year (2026); log price/log miles reduce positive skew. Brand, model,
fuel, tax, MPG and engine displacement remain available to the classifier. IQR outliers
are reported and retained rather than deleting legitimate expensive or high-mileage cars.

**Team task:** write your own observations from each plot and justify these feature choices, as required by the guidelines.
'''),code('''audit_plots(df, train)
display(train.describe(include='all').T)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
features = CarFeatures().transform(X_train)
for ax, col in zip(axes.ravel(), ['brand', 'model', 'fuel_type']):
    features[col].value_counts().head(12).plot.bar(ax=ax)
    ax.set(title=f'{col}: training distribution (top 12)', xlabel=col, ylabel='Cars')
plt.tight_layout(); plt.show()
'''),md('## Model and search space\nThe displayed estimator is the untuned baseline. The search includes its parameter settings. A tied or lower test score after tuning is reported honestly.'),code(f'''slug = '{slug}'
name, estimator, parameter_grid = configurations()[slug]
display(estimator)
display(parameter_grid)
baseline = build_pipeline(clone(estimator))
warning_messages = []
'''),md('## Before tuning: five-fold CV and full training fit'),code('''with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    baseline_cv = cross_validate(baseline, X_train, y_train, cv=CV, scoring=SCORING,
                                 n_jobs=1, error_score='raise')
    baseline.fit(X_train, y_train)
warning_messages.extend(str(w.message) for w in captured)
display(pd.DataFrame({k: v for k, v in baseline_cv.items() if k.startswith('test_')}))
print('Mean baseline macro F1:', baseline_cv['test_f1_macro'].mean())
'''),md('## After tuning: GridSearchCV\nImputation, encoding and scaling are refitted independently inside each fold. The MLP uses a fixed convergence budget without internal early stopping; no test rows are used while fitting.'),code('''search = GridSearchCV(build_pipeline(clone(estimator)), parameter_grid,
                      scoring=SCORING, refit='f1_macro', cv=CV,
                      n_jobs=1, error_score='raise', return_train_score=False)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    search.fit(X_train, y_train)
warning_messages.extend(str(w.message) for w in captured)
tuned = search.best_estimator_
print('Best parameters:', search.best_params_)
print('Best training CV macro F1:', search.best_score_)
display(pd.DataFrame(search.cv_results_)[['params', 'mean_test_f1_macro',
    'std_test_f1_macro', 'mean_test_f1_weighted', 'rank_test_f1_macro']].sort_values('rank_test_f1_macro'))
print('Recorded training warnings:', sorted(set(warning_messages)) or 'None')
'''),md('## Held-out test comparison\nROC-AUC and average precision consistently treat **Automatic as positive**. Weighted recall equals accuracy in this single-label task. CV scores used for tuning are selection estimates, not independent generalization estimates.'),code('''comparison = evaluate_pair(slug, name, baseline, tuned, X_test, y_test,
                           baseline_cv, search, warning_messages)
display(comparison.round(4))
metric_cols = ['Accuracy', 'F1_weighted', 'F1_macro', 'Automatic_recall', 'ROC_AUC', 'CV_macro_F1']
delta = comparison.loc[1, metric_cols] - comparison.loc[0, metric_cols]
display(delta.to_frame('After minus before'))
comparison.set_index('Stage')[['F1_macro', 'F1_weighted', 'Balanced_accuracy', 'ROC_AUC']].T.plot.bar(figsize=(9, 4), ylim=(0, 1))
plt.title('Before and after tuning: held-out metrics'); plt.xlabel('Metric'); plt.ylabel('Score')
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
display(pd.read_csv(ROOT / 'results' / slug / 'after_class_report.csv', index_col=0))
plot_pair(slug, baseline, tuned, X_test, y_test)
'''),md('''## Team interpretation and viva preparation
Complete these prompts yourself after inspecting the actual outputs:
- Which metric changed most, and by how much? Did test performance improve or decline?
- How many Automatic cars were missed before and after tuning?
- Why is oversampling unnecessary here? What does distinct listing data establish, and what does it not prove?
- Explain this model's training mechanism and the selected hyperparameters.
- Why might CV improvement fail to produce test improvement?

Models and numeric results are exported to `models/` and `results/`. The GUI reads those same files.
Source: [scikit-learn](https://scikit-learn.org/stable/). See data/CLASSIFICATION_DATA_SOURCE.md for dataset provenance.
AI-assisted code scaffolding is disclosed in README. Your interpretation must be original.
''')]
    save(f'{i:02d}_{slug}.ipynb',cells)

for i,(kind,title) in enumerate([('kmeans','K-Means'),('agglomerative','Agglomerative Hierarchical Clustering')],11):
    cells=[md(f'''# {title}: baseline and tuned car segments
**Review 2 · Clustering · no target labels used in fitting**

Dataset: 4,009 used-car listings. Price is USD; `milage` contains miles, despite its spelling.
Numeric segmentation uses **car age, log price and log mileage**, standardized after training-only
median imputation. Brand, engine text, transmission, colour, accident and title fields do not define
distance; they remain available for post-fit interpretation. This avoids arbitrary weighting of
high-cardinality categories. Price is a clustering feature, not a supervised target.

Baseline: k=3 (Ward linkage for hierarchical clustering). Search: k=2..6; hierarchical also searches
Ward/complete/average linkage. Fit preprocessing on each training fold, assign validation rows,
and select by mean five-fold validation silhouette. K-Means uses native prediction. Hierarchical
uses an explicit nearest-training-centroid **proxy**, because the algorithm has no native predict.
This proxy evaluates deployable assignments, not a new hierarchical fit on validation cars.

A separate 20% holdout is never used to select k/linkage. Report intrinsic training partition metrics
and held-out assignment metrics separately. Silhouette is sampled to at most 2,000 rows with seed 42;
Davies–Bouldin and Calinski–Harabasz use all rows in each evaluated set.
'''),code(BOOT+'''from src.clustering import *
raw, df, train, test = load_data()
display(raw.dtypes.to_frame('dtype').astype(str))
display(raw.isna().sum().to_frame('Missing raw values'))
print('Raw / cleaned / train / holdout:', len(raw), len(df), len(train), len(test))
print('Exact duplicates removed:', len(raw)-len(df))
display(train[['price_usd','mileage_miles','car_age']].describe())
'''),md('## Training EDA and preparation\nMoney and mileage strings are parsed to numbers; invalid values become missing. Log transforms reduce skew while retaining extreme cars. Missing values are learned from each training fold.\n\n**Team task:** add observations and justify the chosen features after reviewing these plots.'),code('''train[['price_usd','mileage_miles','car_age']].hist(bins=30,figsize=(11,3))
plt.suptitle('Training numeric distributions'); plt.tight_layout(); plt.show()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(train.mileage_miles, train.price_usd, alpha=.3, s=9)
axes[0].set(xlabel='Mileage (miles)',ylabel='Price (USD)',title='Price vs mileage')
axes[1].scatter(train.car_age, train.price_usd, alpha=.3, s=9)
axes[1].set(xlabel='Age (years, reference 2026)',ylabel='Price (USD)',title='Price vs age')
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(ClusterFeatures().transform(train).corr(),annot=True,cmap='coolwarm',ax=ax)
ax.set_title('Training clustering-feature correlations'); plt.tight_layout(); plt.show()
q = train[['price_usd','mileage_miles','car_age']]
iqr=q.quantile(.75)-q.quantile(.25)
display(((q<q.quantile(.25)-1.5*iqr)|(q>q.quantile(.75)+1.5*iqr)).sum().to_frame('IQR outliers retained'))
'''),md('## Five-fold parameter search\nCandidates with collapsed validation clusters are rejected. Higher silhouette and Calinski–Harabasz are better; lower Davies–Bouldin is better. The metrics may disagree.'),code(f'''kind = '{kind}'
search_summary, fold_results, best = tune(kind, train)
display(search_summary.sort_values('CV_silhouette', ascending=False))
print('Chosen on training folds only:', best)
out = ROOT / 'results' / kind
out.mkdir(parents=True, exist_ok=True)
fold_results.to_csv(out / 'cv_fold_results.csv', index=False)
'''),md('## Before/after comparison and held-out checks'),code('''comparison, fitted, prep, X, X_holdout, pca, profiles = fit_compare(kind, train, test, search_summary, best)
display(comparison.round(4))
cols=['Silhouette','Davies_Bouldin','Calinski_Harabasz','CV_silhouette','Holdout_Silhouette']
display((comparison.loc[1,cols]-comparison.loc[0,cols]).to_frame('After minus before'))
display(profiles)
'''),md('## Required visualizations\nBoth fits use the same PCA axes. PCA and t-SNE are visualizations only, not inputs to model fitting. Cluster numbers are arbitrary, not quality rankings. The elbow/dendrogram is a diagnostic; numerical selection comes from the stated CV rule.'),code('''plot_clusters(kind, fitted, X, pca)
display(pd.read_csv(out / 'fuel_type_profile.csv', index_col=0).round(3))
'''),md('''## Team interpretation
Write your own conclusions: which cluster contains older/high-mileage cars, how price differs,
whether the elbow or dendrogram supports the selected k, and whether all metrics agree.
Discuss limitations of three numeric features, centroid-proxy assignment (hierarchical only),
sampled silhouette and t-SNE. Do not call these clusters known ground-truth categories.

Source: [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html).
''')]
    save(f'{i:02d}_{kind}.ipynb',cells)

cells=[md('''# Review 2: consolidated comparison of all ten classifiers
The guidelines require all ten algorithms on a common held-out split. This notebook reruns the five
Part A baselines using exactly the same preparation and split as Part B. These are **new comparable
runs**, not copied Review 1 scores. Part A uses fixed baseline settings; Part B includes baseline and tuned settings.

Run notebooks 06–10 first. Select the deployment model using training CV macro F1 only.
The test table is descriptive; no hyperparameters or winner are chosen using test scores.
'''),code(BOOT+CLASS_IMPORT),code('''import joblib
rows=[]
for slug, (name, estimator) in part_a_models().items():
    model=build_pipeline(estimator)
    cv=cross_validate(model,X_train,y_train,cv=CV,scoring=SCORING,n_jobs=1,error_score='raise')
    model.fit(X_train,y_train)
    row={'Slug':slug,'Model':name,'Part':'A rerun','Stage':'Baseline',**metrics(model,X_test,y_test),
         'CV_macro_F1':cv['test_f1_macro'].mean(),'CV_macro_F1_std':cv['test_f1_macro'].std()}
    rows.append(row)
    joblib.dump(model,ROOT/'models'/f'{slug}_before.joblib',compress=3)
    print('Completed:',name)
part_a=pd.DataFrame(rows)
part_a.to_csv(ROOT/'results/part_a_rerun.csv',index=False)
display(part_a.round(4))
'''),code('''part_b=[]
for slug in configurations():
    path=ROOT/'results'/slug/'comparison.csv'
    if not path.exists(): raise FileNotFoundError(f'Run the {slug} notebook first.')
    t=pd.read_csv(path);t['Slug']=slug;t['Part']='B'
    part_b.append(t)
part_b=pd.concat(part_b,ignore_index=True)
all_runs=pd.concat([part_a,part_b],ignore_index=True)
all_runs.to_csv(ROOT/'results/all_classification_runs.csv',index=False)
part_b.to_csv(ROOT/'results/part_b_before_after.csv',index=False)
after=part_b[part_b.Stage=='After tuning'].sort_values('CV_macro_F1',ascending=False)
after.to_csv(ROOT/'results/after_tuning_comparison.csv',index=False)
print('All five Part B models AFTER tuning (ordered by training CV macro F1):')
display(after.round(4))
fig,ax=plt.subplots(figsize=(9,4))
after.set_index('Model')[['F1_macro','F1_weighted','Accuracy','ROC_AUC']].plot.bar(ax=ax,ylim=(0,1))
ax.set(title='After tuning: all five Part B classifiers',ylabel='Held-out score',xlabel='Model')
plt.xticks(rotation=15);plt.tight_layout();plt.savefig(ROOT/'results/after_tuning_comparison.png',dpi=130);plt.show()
delta_rows=[]
for slug,g in part_b.groupby('Slug',sort=False):
    before=g[g.Stage=='Before tuning'].iloc[0];tuned_row=g[g.Stage=='After tuning'].iloc[0]
    cols=['Accuracy','Precision_weighted','Recall_weighted','F1_weighted','F1_macro','ROC_AUC','CV_macro_F1']
    delta_rows.append({'Model':before.Model,**{col:float(tuned_row[col]-before[col]) for col in cols}})
deltas=pd.DataFrame(delta_rows)
deltas.to_csv(ROOT/'results/tuning_deltas.csv',index=False)
print('After minus before:');display(deltas.round(4))

# Choose each model's baseline/tuned version and the global winner on training CV only.
selected=all_runs.sort_values('CV_macro_F1',ascending=False,kind='stable').drop_duplicates('Slug')
selected.to_csv(ROOT/'results/ten_model_comparison.csv',index=False)
display(selected.round(4))
winner=selected.iloc[0]
suffix='after' if winner.Stage=='After tuning' else 'before'
selection={'slug':winner.Slug,'name':winner.Model,'stage':winner.Stage,
           'model_file':f'{winner.Slug}_{suffix}.joblib','cv_macro_f1':float(winner.CV_macro_F1),
           'selection_basis':'Highest five-fold training CV macro F1; test metrics not used'}
(ROOT/'results/selected_model.json').write_text(json.dumps(selection,indent=2))
print('Deployment selection:',selection)
ax=selected.set_index('Model')[['F1_macro','F1_weighted','ROC_AUC']].plot.barh(figsize=(10,6),xlim=(0,1))
ax.set(title='Ten-model held-out comparison',xlabel='Score',ylabel='Model')
plt.tight_layout();plt.savefig(ROOT/'results/ten_model_comparison.png',dpi=130);plt.show()
'''),md('''## Complete your review discussion
Explain why the selected model was chosen using CV, compare its held-out performance to alternatives,
and discuss the Automatic-class precision/recall tradeoff. If tuning did not improve a metric, report
that outcome rather than changing the test split. Five-fold CV covers all ten models, exceeding the
two-model minimum. Review 1 regression is outside this package because its notebook was not supplied.
''')]
save('13_all_models_comparison.ipynb',cells)
print('Created eight notebooks.')
