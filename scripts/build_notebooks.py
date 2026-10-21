"""Regenerate the project notebooks. Run an execution script to populate outputs."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def markdown(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


def save(name, cells):
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    nbf.write(notebook, ROOT / "notebooks" / name)


BOOTSTRAP = '''from pathlib import Path
import json
import os
import sys
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
if not (ROOT / "src").exists():
    raise RuntimeError("Open this notebook inside the project or project/notebooks folder.")
sys.path.insert(0, str(ROOT))

from threadpoolctl import threadpool_limits

thread_limit = threadpool_limits(limits=1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

%matplotlib inline
sns.set_theme(style="whitegrid", palette="colorblind")
'''


CLASSIFICATION_IMPORTS = '''from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import GridSearchCV, cross_validate
import joblib

from src.classification import *
'''


CLASSIFICATION_DATA = '''raw, df, train, test = load_data()
X_train, y_train = train[RAW_FEATURES], train[TARGET]
X_test, y_test = test[RAW_FEATURES], test[TARGET]

display(
    pd.DataFrame(
        {
            "Raw": raw[TARGET].value_counts(),
            "Cleaned": df[TARGET].value_counts(),
            "Train": y_train.value_counts(),
            "Test": y_test.value_counts(),
        }
    )
)
display(df.dtypes.to_frame("dtype").astype(str))
print("Raw shape:", raw.shape)
print("Cleaned shape:", df.shape)
print("Removed exact duplicate rows:", len(raw) - len(df))
print("Training rows:", len(train), "| Held-out test rows:", len(test))
assert not df.duplicated().any()
assert set(train.index).isdisjoint(test.index)
'''


CLASSIFICATION_EDA = '''audit_plots(df, train)
display(train.describe(include="all").T)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
engineered_train = CarFeatures().transform(X_train)
for axis, column in zip(axes.ravel(), ["brand", "model", "fuel_type"]):
    engineered_train[column].value_counts().head(12).plot.bar(ax=axis)
    axis.set(
        title=f"{column}: training distribution (top 12)",
        xlabel=column,
        ylabel="Cars",
    )
plt.tight_layout()
plt.show()
'''


PART_A_NOTEBOOKS = [
    (
        "01_logistic_regression.ipynb",
        "logistic_regression",
        "Logistic Regression",
        "Logistic Regression estimates class probabilities and exposes signed coefficients that help explain feature influence.",
    ),
    (
        "02_knn_classifier.ipynb",
        "knn",
        "K-Nearest Neighbors Classifier",
        "KNN predicts from nearby training examples, so scaling is essential because feature distances determine the neighbors.",
    ),
    (
        "03_gaussian_naive_bayes.ipynb",
        "naive_bayes",
        "Gaussian Naive Bayes",
        "Gaussian Naive Bayes estimates class probabilities under conditional-independence and Gaussian feature assumptions.",
    ),
    (
        "04_decision_tree_classifier.ipynb",
        "decision_tree",
        "Decision Tree Classifier",
        "The Decision Tree learns feature-based splits; its depth and leaf structure determine model complexity.",
    ),
    (
        "05_svc.ipynb",
        "svc",
        "Support Vector Classifier",
        "SVC learns a decision boundary between the two classes, and scaling is important because the boundary depends on feature magnitudes.",
    ),
]


for filename, slug, title, model_observation in PART_A_NOTEBOOKS:
    cells = [
        markdown(
            f'''# {title}

**Review 2 - Classification Part A - target: transmission**

This project-native rerun uses the same balanced 2,500-row UK dataset, fixed 80:20 split,
feature engineering and fold-safe preprocessing as every other classifier in TORQUEX.
The original Review 1 score is not reused. Part A keeps the established baseline settings;
five-fold cross-validation evaluates the model but does not tune it.'''
        ),
        markdown("## 1. Imports"),
        code(BOOTSTRAP + CLASSIFICATION_IMPORTS),
        markdown("## 2. Dataset audit, target definition and preprocessing"),
        code(CLASSIFICATION_DATA),
        markdown("## 3. Required EDA"),
        code(CLASSIFICATION_EDA),
        markdown(
            "## 4. Model training and tuning\n\n"
            "Part A is intentionally a baseline rerun, so no hyperparameter search is performed. "
            "The complete pipeline is still evaluated with five-fold stratified cross-validation."
        ),
        code(
            f'''slug = "{slug}"
name, estimator = part_a_models()[slug]
model = build_pipeline(clone(estimator))

print("Baseline estimator:")
display(estimator)
print("No parameter grid: this is the comparable Part A baseline rerun.")'''
        ),
        code(
            '''with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    cv_result = cross_validate(
        model,
        X_train,
        y_train,
        cv=CV,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    model.fit(X_train, y_train)

warning_messages = sorted({str(item.message) for item in captured})
display(pd.DataFrame({key: value for key, value in cv_result.items() if key.startswith("test_")}))
print("Mean training CV macro F1:", cv_result["test_f1_macro"].mean())
print("Recorded training warnings:", warning_messages or "None")'''
        ),
        markdown("## 5. Evaluation"),
        code(
            '''result = pd.DataFrame(
    [
        {
            "Slug": slug,
            "Model": name,
            "Part": "A rerun",
            "Stage": "Baseline",
            **metrics(model, X_test, y_test),
            "CV_macro_F1": float(cv_result["test_f1_macro"].mean()),
            "CV_macro_F1_std": float(cv_result["test_f1_macro"].std()),
        }
    ]
)
display(result.round(4))
display(
    pd.DataFrame(
        classification_report(
            y_test,
            model.predict(X_test),
            output_dict=True,
            zero_division=0,
        )
    ).T.round(4)
)

output_dir = ROOT / "results" / slug
output_dir.mkdir(parents=True, exist_ok=True)
result.to_csv(output_dir / "comparison.csv", index=False)
pd.DataFrame(
    {key: value for key, value in cv_result.items() if key.startswith("test_")}
).to_csv(output_dir / "baseline_cv_folds.csv", index=False)
pd.DataFrame(
    classification_report(
        y_test,
        model.predict(X_test),
        output_dict=True,
        zero_division=0,
    )
).T.to_csv(output_dir / "before_class_report.csv")
joblib.dump(model, ROOT / "models" / f"{slug}_before.joblib", compress=3)

metadata = {
    "selection_metric": "Five-fold training CV macro F1 (evaluation only; no Part A tuning)",
    "positive_class": "Automatic",
    "warnings": warning_messages,
    "test_used_for_tuning": False,
}
(output_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2, default=str), encoding="utf-8"
)'''
        ),
        markdown("## 6. Confusion matrix and model interpretation"),
        code(
            '''fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test,
    labels=["Automatic", "Manual"],
    display_labels=["Automatic", "Manual"],
    cmap="Blues",
    colorbar=False,
    ax=axes[0],
)
axes[0].set_title("Held-out confusion matrix")

automatic_scores = automatic_probability(model, X_test)
automatic_target = (y_test == "Automatic").astype(int)
RocCurveDisplay.from_predictions(
    automatic_target, automatic_scores, name=name, ax=axes[1]
)
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_title("ROC - Automatic is positive")

PrecisionRecallDisplay.from_predictions(
    automatic_target, automatic_scores, name=name, ax=axes[2]
)
axes[2].axhline(
    automatic_target.mean(), linestyle="--", color="gray", label="Automatic prevalence"
)
axes[2].legend()
axes[2].set_title("Precision-recall - Automatic")
plt.tight_layout()
plt.show()

fitted_estimator = model.named_steps["model"]
feature_names = model.named_steps["preprocess"].get_feature_names_out()
if hasattr(fitted_estimator, "feature_importances_"):
    importance = pd.Series(
        fitted_estimator.feature_importances_, index=feature_names
    ).nlargest(15).sort_values()
    importance.plot.barh(figsize=(8, 5), color="#128276")
    plt.title("Top feature importances")
    plt.xlabel("Impurity-based importance")
    plt.tight_layout()
    plt.show()
elif hasattr(fitted_estimator, "coef_"):
    coefficients = pd.Series(fitted_estimator.coef_[0], index=feature_names)
    strongest = coefficients.loc[
        coefficients.abs().nlargest(15).index
    ].sort_values()
    strongest.plot.barh(figsize=(8, 5), color="#128276")
    plt.title("Strongest signed coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()
else:
    print("This estimator does not expose direct coefficients or feature importances.")'''
        ),
        markdown(
            f'''## 7. Observation / Interpretation

**Observation:** This notebook uses **{title}** to predict `transmission`
(Automatic or Manual) from the available car features.

- Cleaning, feature engineering and the fixed split are shared by all ten classifiers.
- Numeric and categorical preprocessing is fitted inside each training fold to avoid leakage.
- Five-fold macro F1 evaluates the Part A baseline; the held-out test set is used only for final reporting.
- Accuracy, weighted precision/recall/F1, macro F1, balanced accuracy, ROC-AUC and the confusion matrix are reported.
- **Model-specific observation:** {model_observation}

**Team task:** Discuss the actual metric values, class-specific errors and limitations visible in the executed output. Do not assume a performance value that is not displayed.'''
        ),
    ]
    save(filename, cells)


PART_B_NOTEBOOKS = [
    ("06_random_forest.ipynb", "random_forest", "Random Forest"),
    ("07_adaboost.ipynb", "adaboost", "AdaBoost"),
    ("08_gradient_boosting.ipynb", "gradient_boosting", "Gradient Boosting"),
    ("09_bagging.ipynb", "bagging", "Bagging"),
    ("10_mlp.ipynb", "mlp", "MLP Neural Network"),
]


for filename, slug, title in PART_B_NOTEBOOKS:
    cells = [
        markdown(
            f'''# {title}

**Review 2 - Classification Part B - target: transmission**

Predict Automatic vs Manual using 2,500 balanced UK listing records. The fixed 80:20
stratified split gives 2,000 training rows and 500 held-out test rows. Selection uses
five-fold training macro F1; the test set never chooses parameters. Preprocessing is fitted
inside each fold, and before/after tuning results use the same features and split.'''
        ),
        markdown("## 1. Imports"),
        code(BOOTSTRAP + CLASSIFICATION_IMPORTS),
        markdown("## 2. Dataset audit, target definition and preprocessing"),
        code(CLASSIFICATION_DATA),
        markdown("## 3. Required EDA"),
        code(CLASSIFICATION_EDA),
        markdown(
            "## 4. Model training and tuning\n\n"
            "The first cell evaluates the untuned baseline. The second uses GridSearchCV; "
            "the search includes the baseline settings, and lower held-out performance is reported honestly."
        ),
        code(
            f'''slug = "{slug}"
name, estimator, parameter_grid = configurations()[slug]
print("Baseline estimator:")
display(estimator)
print("Parameter grid:")
display(parameter_grid)

baseline = build_pipeline(clone(estimator))
warning_messages = []
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    baseline_cv = cross_validate(
        baseline,
        X_train,
        y_train,
        cv=CV,
        scoring=SCORING,
        n_jobs=1,
        error_score="raise",
    )
    baseline.fit(X_train, y_train)

warning_messages.extend(str(item.message) for item in captured)
display(
    pd.DataFrame(
        {{key: value for key, value in baseline_cv.items() if key.startswith("test_")}}
    )
)
print("Mean baseline CV macro F1:", baseline_cv["test_f1_macro"].mean())'''
        ),
        code(
            '''search = GridSearchCV(
    build_pipeline(clone(estimator)),
    parameter_grid,
    scoring=SCORING,
    refit="f1_macro",
    cv=CV,
    n_jobs=1,
    error_score="raise",
    return_train_score=False,
)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter("always")
    search.fit(X_train, y_train)

warning_messages.extend(str(item.message) for item in captured)
tuned = search.best_estimator_
print("Best parameters:", search.best_params_)
print("Best training CV macro F1:", search.best_score_)
display(
    pd.DataFrame(search.cv_results_)[
        [
            "params",
            "mean_test_f1_macro",
            "std_test_f1_macro",
            "mean_test_f1_weighted",
            "rank_test_f1_macro",
        ]
    ].sort_values("rank_test_f1_macro")
)
print("Recorded training warnings:", sorted(set(warning_messages)) or "None")'''
        ),
        markdown("## 5. Evaluation"),
        code(
            '''comparison = evaluate_pair(
    slug,
    name,
    baseline,
    tuned,
    X_test,
    y_test,
    baseline_cv,
    search,
    warning_messages,
)
display(comparison.round(4))

metric_columns = [
    "Accuracy",
    "F1_weighted",
    "F1_macro",
    "Automatic_recall",
    "ROC_AUC",
    "CV_macro_F1",
]
delta = comparison.loc[1, metric_columns] - comparison.loc[0, metric_columns]
display(delta.to_frame("After minus before"))

comparison.set_index("Stage")[
    ["F1_macro", "F1_weighted", "Balanced_accuracy", "ROC_AUC"]
].T.plot.bar(figsize=(9, 4), ylim=(0, 1))
plt.title("Before and after tuning: held-out metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()'''
        ),
        markdown("## 6. Confusion matrix and model interpretation"),
        code(
            '''display(
    pd.read_csv(
        ROOT / "results" / slug / "after_class_report.csv", index_col=0
    ).round(4)
)
plot_pair(slug, baseline, tuned, X_test, y_test)'''
        ),
        markdown(
            f'''## 7. Observation / Interpretation

**Observation:** This notebook compares the baseline and CV-selected **{title}** classifier on one untouched test split.

- Identify the metric that changed most and report the exact after-minus-before value.
- Compare Automatic-class errors in the two confusion matrices.
- Explain how the selected hyperparameters affect this model's learning mechanism.
- Explain why a CV improvement may not produce a held-out improvement.
- Discuss why oversampling is unnecessary for this balanced dataset.

Models and numeric results are exported to `models/` and `results/`; the GUI reads those same files.
Complete the interpretation using the executed evidence and your own wording.'''
        ),
    ]
    save(filename, cells)


CLUSTERING_NOTEBOOKS = [
    ("11_kmeans.ipynb", "kmeans", "K-Means"),
    ("12_agglomerative.ipynb", "agglomerative", "Agglomerative Hierarchical Clustering"),
]


for filename, kind, title in CLUSTERING_NOTEBOOKS:
    cells = [
        markdown(
            f'''# {title}

**Review 2 - Clustering - no target labels used in fitting**

The 4,009-row used-car dataset is segmented using car age, log price and log mileage.
Training-only preprocessing and five-fold validation choose the model configuration; a separate
20% holdout checks assignment quality. PCA and t-SNE are visualizations, not fitting inputs.'''
        ),
        markdown("## 1. Imports"),
        code(BOOTSTRAP + "from src.clustering import *\n"),
        markdown("## 2. Dataset audit and preprocessing"),
        code(
            '''raw, df, train, test = load_data()
display(raw.dtypes.to_frame("dtype").astype(str))
display(raw.isna().sum().to_frame("Missing raw values"))
print("Raw / cleaned / train / holdout:", len(raw), len(df), len(train), len(test))
print("Exact duplicates removed:", len(raw) - len(df))
display(train[["price_usd", "mileage_miles", "car_age"]].describe())
assert set(train.index).isdisjoint(test.index)'''
        ),
        markdown("## 3. Required EDA"),
        code(
            '''train[["price_usd", "mileage_miles", "car_age"]].hist(
    bins=30, figsize=(11, 3)
)
plt.suptitle("Training numeric distributions")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(train.mileage_miles, train.price_usd, alpha=0.3, s=9)
axes[0].set(
    xlabel="Mileage (miles)", ylabel="Price (USD)", title="Price vs mileage"
)
axes[1].scatter(train.car_age, train.price_usd, alpha=0.3, s=9)
axes[1].set(
    xlabel="Age (years, reference 2026)",
    ylabel="Price (USD)",
    title="Price vs age",
)
plt.tight_layout()
plt.show()

fig, axis = plt.subplots(figsize=(5, 4))
sns.heatmap(
    ClusterFeatures().transform(train).corr(), annot=True, cmap="coolwarm", ax=axis
)
axis.set_title("Training clustering-feature correlations")
plt.tight_layout()
plt.show()

numeric = train[["price_usd", "mileage_miles", "car_age"]]
iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
outlier_counts = (
    (numeric < numeric.quantile(0.25) - 1.5 * iqr)
    | (numeric > numeric.quantile(0.75) + 1.5 * iqr)
).sum()
display(outlier_counts.to_frame("IQR outliers retained"))'''
        ),
        markdown(
            "## 4. Model training and tuning\n\n"
            "Candidates are selected using mean five-fold validation silhouette. Higher silhouette "
            "and Calinski-Harabasz are better; lower Davies-Bouldin is better."
        ),
        code(
            f'''kind = "{kind}"
if kind == "kmeans":
    print("Baseline: k=3 | Search: k=2 through 6")
    print("Validation assignment: native K-Means prediction")
else:
    print("Baseline: k=3 with Ward linkage")
    print("Search: k=2 through 6 with Ward, complete and average linkage")
    print("Validation assignment: nearest-training-centroid proxy")'''
        ),
        code(
            '''search_summary, fold_results, best = tune(kind, train)
display(search_summary.sort_values("CV_silhouette", ascending=False))
print("Chosen on training folds only:", best)

output_dir = ROOT / "results" / kind
output_dir.mkdir(parents=True, exist_ok=True)
fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)'''
        ),
        markdown("## 5. Evaluation"),
        code(
            '''comparison, fitted, prep, X, X_holdout, pca, profiles = fit_compare(
    kind, train, test, search_summary, best
)
display(comparison.round(4))

metric_columns = [
    "Silhouette",
    "Davies_Bouldin",
    "Calinski_Harabasz",
    "CV_silhouette",
    "Holdout_Silhouette",
]
display(
    (comparison.loc[1, metric_columns] - comparison.loc[0, metric_columns]).to_frame(
        "After minus before"
    )
)
display(profiles)'''
        ),
        markdown("## 6. Required visualizations and cluster interpretation"),
        code(
            '''plot_clusters(kind, fitted, X, pca)
display(
    pd.read_csv(output_dir / "fuel_type_profile.csv", index_col=0).round(3)
)'''
        ),
        markdown(
            '''## 7. Observation / Interpretation

Complete these observations using the executed evidence:

- Describe the age, mileage and price profile of each cluster.
- Explain whether the elbow or dendrogram supports the selected number of clusters.
- Compare silhouette, Davies-Bouldin and Calinski-Harabasz before and after tuning.
- Discuss the three-feature limitation and the arbitrary nature of cluster numbers.
- For hierarchical clustering, explain the centroid-proxy limitation for unseen cars.
- Treat t-SNE as a visualization only and do not describe clusters as known ground-truth classes.'''
        ),
    ]
    save(filename, cells)


comparison_cells = [
    markdown(
        '''# All Models Comparison

**Review 2 - Consolidated comparison of all ten classifiers**

The five Part A baselines and five Part B classifiers use the same dataset, split, features and
fold-safe preprocessing. Model selection uses five-fold training macro F1 only; held-out metrics
describe final performance and never choose the winner.'''
    ),
    markdown("## 1. Imports"),
    code(BOOTSTRAP + CLASSIFICATION_IMPORTS),
    markdown("## 2. Dataset audit and comparison scope"),
    code(
        CLASSIFICATION_DATA
        + '''\nexpected_models = pd.DataFrame(
    [
        {"Slug": slug, "Model": name, "Part": "A"}
        for slug, (name, _) in part_a_models().items()
    ]
    + [
        {"Slug": slug, "Model": values[0], "Part": "B"}
        for slug, values in configurations().items()
    ]
)
display(expected_models)
assert len(expected_models) == 10
'''
    ),
    markdown("## 3. Required result audit"),
    code(
        '''part_a_tables = []
for slug in part_a_models():
    path = ROOT / "results" / slug / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run the {slug} notebook first: {path}")
    table = pd.read_csv(path)
    part_a_tables.append(table)
part_a = pd.concat(part_a_tables, ignore_index=True)
part_a.to_csv(ROOT / "results" / "part_a_rerun.csv", index=False)

part_b_tables = []
for slug in configurations():
    path = ROOT / "results" / slug / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run the {slug} notebook first: {path}")
    table = pd.read_csv(path)
    table["Slug"] = slug
    table["Part"] = "B"
    part_b_tables.append(table)
part_b = pd.concat(part_b_tables, ignore_index=True)

display(part_a[["Model", "Stage", "CV_macro_F1", "F1_macro"]].round(4))
display(
    part_b[["Model", "Stage", "CV_macro_F1", "F1_macro"]]
    .sort_values(["Model", "Stage"])
    .round(4)
)
assert part_a["Slug"].nunique() == 5
assert part_b["Slug"].nunique() == 5'''
    ),
    markdown("## 4. Model result consolidation"),
    code(
        '''all_runs = pd.concat([part_a, part_b], ignore_index=True)
all_runs.to_csv(ROOT / "results" / "all_classification_runs.csv", index=False)
part_b.to_csv(ROOT / "results" / "part_b_before_after.csv", index=False)

after = part_b[part_b.Stage == "After tuning"].sort_values(
    "CV_macro_F1", ascending=False
)
after.to_csv(ROOT / "results" / "after_tuning_comparison.csv", index=False)
print("All five Part B models after tuning, ordered by training CV macro F1:")
display(after.round(4))'''
    ),
    code(
        '''delta_rows = []
for slug, group in part_b.groupby("Slug", sort=False):
    before = group[group.Stage == "Before tuning"].iloc[0]
    tuned_row = group[group.Stage == "After tuning"].iloc[0]
    columns = [
        "Accuracy",
        "Precision_weighted",
        "Recall_weighted",
        "F1_weighted",
        "F1_macro",
        "ROC_AUC",
        "CV_macro_F1",
    ]
    delta_rows.append(
        {
            "Model": before.Model,
            **{column: float(tuned_row[column] - before[column]) for column in columns},
        }
    )
deltas = pd.DataFrame(delta_rows)
deltas.to_csv(ROOT / "results" / "tuning_deltas.csv", index=False)
print("After minus before:")
display(deltas.round(4))'''
    ),
    markdown("## 5. Evaluation and final model selection"),
    code(
        '''ranked_runs = all_runs.assign(
    _stage_priority=all_runs["Stage"].eq("After tuning").astype(int)
)
selected = (
    ranked_runs.sort_values(
        ["CV_macro_F1", "_stage_priority"],
        ascending=[False, False],
        kind="stable",
    )
    .drop_duplicates("Slug")
    .drop(columns="_stage_priority")
    .reset_index(drop=True)
)
selected.to_csv(ROOT / "results" / "ten_model_comparison.csv", index=False)
display(selected.round(4))

winner = selected.iloc[0]
suffix = "after" if winner.Stage == "After tuning" else "before"
selection = {
    "slug": winner.Slug,
    "name": winner.Model,
    "stage": winner.Stage,
    "model_file": f"{winner.Slug}_{suffix}.joblib",
    "cv_macro_f1": float(winner.CV_macro_F1),
    "selection_basis": "Highest five-fold training CV macro F1; exact ties prefer the CV-selected after-tuning variant; test metrics not used",
}
(ROOT / "results" / "selected_model.json").write_text(
    json.dumps(selection, indent=2), encoding="utf-8"
)
print("Final model selection:", selection)'''
    ),
    markdown("## 6. Comparison visualizations and model interpretation"),
    code(
        '''fig, axis = plt.subplots(figsize=(9, 4))
after.set_index("Model")[["F1_macro", "F1_weighted", "Accuracy", "ROC_AUC"]].plot.bar(
    ax=axis, ylim=(0, 1)
)
axis.set(
    title="After tuning: all five Part B classifiers",
    ylabel="Held-out score",
    xlabel="Model",
)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(ROOT / "results" / "after_tuning_comparison.png", dpi=130)
plt.show()

axis = selected.set_index("Model")[["F1_macro", "F1_weighted", "ROC_AUC"]].plot.barh(
    figsize=(10, 6), xlim=(0, 1)
)
axis.set(title="Ten-model held-out comparison", xlabel="Score", ylabel="Model")
plt.tight_layout()
plt.savefig(ROOT / "results" / "ten_model_comparison.png", dpi=130)
plt.show()'''
    ),
    markdown(
        '''## 7. Observation / Interpretation

Complete the review discussion using the tables and plots above:

- Explain why the selected model wins according to training CV macro F1.
- Compare its held-out performance with the strongest alternatives.
- Discuss the Automatic-class precision/recall tradeoff.
- Report any metric that did not improve after tuning instead of changing the split.
- Explain why the common split and preprocessing are necessary for a fair ten-model comparison.

The original Review 1 scores are not mixed into this table; all ten values are comparable reruns.'''
    ),
]
save("13_all_models_comparison.ipynb", comparison_cells)

print("Created thirteen notebooks in the shared seven-section format.")
