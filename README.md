# TORQUEX — Automotive Intelligence

Review 2 classification and clustering, with a red automotive Streamlit interface, executed notebooks, saved models and reproducible comparisons. Open the app immediately using the included trained artifacts; retraining is optional.

## Start the GUI

1. Extract the entire ZIP; keep the folder structure intact.
2. Install **Python 3.12** if needed.
3. On Windows, double-click **START_TORQUEX.bat**. On macOS/Linux, open a terminal in this folder and run `sh start_torquex.sh`.
4. Leave the terminal open. Streamlit opens the browser at `http://localhost:8501`.

The first launch downloads dependencies and may take several minutes. Later launches reuse the environment and trained models. Internet is needed for dependency installation, not for normal local use. The Windows launcher requires the Python `py` launcher; otherwise use the manual setup below with your Python 3.12 executable.

Manual setup, from this folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Use the pinned dependency versions, especially scikit-learn, with the supplied model files. Stop the server with Ctrl+C. If port 8501 is occupied, append `--server.port 8502` to the Streamlit command.

## Problem statement and datasets

TORQUEX addresses three automotive machine-learning tasks: predicting vehicle prices with regression, predicting Manual versus Automatic transmission with classification, and discovering market segments with clustering.

The new balanced dataset is used for **classification**. The original `used_cars.csv` is retained for **clustering**.

| Track | File | Rows | Target / inputs | Units |
|---|---|---:|---|---|
| Classification | `data/classification_cars.csv` | 2,500 | Manual or Automatic; nine predictors | GBP, miles, litres |
| Clustering | `data/used_cars.csv` | 4,009 | Age, log price, log mileage | USD, miles |

Transmission is a categorical target, so this is classification, not regression. The classification CSV has exactly **1,250 Manual and 1,250 Automatic** records, no missing values, no duplicate records and no duplicate predictor rows. No synthetic rows or repeated oversampling copies are used. These are distinct real listings from a public historical UK used-car dataset; without VINs, they cannot be certified as different physical vehicles. The balanced sample does not estimate natural market transmission prevalence.

Source: [Aditya Desai's UK Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes). Source-row provenance, selection audit and file hashes are included under `data/`. Read `data/CLASSIFICATION_DATA_SOURCE.md` for sampling and source details.

Clustering source: [the original project CSV](https://github.com/rohithtej2109/Smart-car-deals/blob/main/used_cars.csv). Its bytes are unchanged from the preceding project package. The two datasets are not merged; keep their currencies and schemas separate.

## What is included

| Notebook | Contents |
|---|---|
| `01_logistic_regression.ipynb` | Logistic Regression baseline, five-fold CV, metrics, confusion matrix, ROC/PR and coefficients |
| `02_knn_classifier.ipynb` | KNN baseline, five-fold CV, metrics, confusion matrix and ROC/PR |
| `03_gaussian_naive_bayes.ipynb` | Gaussian Naive Bayes baseline, five-fold CV, metrics, confusion matrix and ROC/PR |
| `04_decision_tree_classifier.ipynb` | Decision Tree baseline, five-fold CV, metrics, confusion matrix, ROC/PR and importance |
| `05_svc.ipynb` | SVC baseline, five-fold CV, metrics, confusion matrix and ROC/PR |
| `06_random_forest.ipynb` | Random Forest baseline, tuning, metrics, confusion matrices, ROC/PR and importance |
| `07_adaboost.ipynb` | AdaBoost baseline, tuning and comparisons |
| `08_gradient_boosting.ipynb` | Gradient Boosting baseline, tuning and comparisons |
| `09_bagging.ipynb` | Bagging with Decision Trees, baseline and tuning |
| `10_mlp.ipynb` | MLP baseline, tuning and training-loss curves |
| `11_kmeans.ipynb` | K-Means before/after, elbow, CV, metrics, PCA and t-SNE |
| `12_agglomerative.ipynb` | Hierarchical clustering, linkage search, dendrogram, metrics, PCA and t-SNE |
| `13_all_models_comparison.ipynb` | Part A baseline results, five-model tuning deltas, all-after-tuning and all-ten comparisons |

All 13 notebooks contain executed outputs and follow the same numbered seven-section format. The original five Part A classifiers—Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree and SVC—are rerun in notebooks 01–05 on the new dataset and consolidated in notebook 13, so the ten-model comparison uses one shared split. The original Review 1 scores are not reused. Part A is not retuned here.

The `models/` directory contains 15 classifier variants and four clustering variants. `src/` holds the actual fitting and plotting functions imported by the readable notebooks; submit it with the notebooks. Every notebook includes prompts for the team's own interpretation.

## GUI walkthrough

- **Dashboard:** custom red GT car artwork, dataset counts, selected model and after-tuning scores.
- **Predict transmission:** choose any classifier, submit one car, optionally compare its Part B baseline and tuned prediction, or upload a CSV and download all predictions. Use the included input template. Batch limit: 10,000 rows.
- **Model comparisons:** five Part B before/after scores and deltas; a separate all-after-tuning view; consolidated ten-model table; model diagnostics and search results.
- **Car clusters:** interactive PCA scatter, cluster profiles, filtered listing downloads, before/after and after-only comparisons, elbow/dendrogram/t-SNE, and new-car segment assignment.
- **Evaluation metrics:** regression, classification and clustering metrics with per-model validation details and downloadable result tables.

Prediction probabilities are uncalibrated model outputs. Inputs outside the training distribution can be unreliable. Changing the selected model does not retrain anything. For hierarchical clustering, new-car assignment uses the nearest training-cluster centroid; it is explicitly labelled as an approximation because the algorithm has no native predict method.

## Experimental design

Classification uses a reproducible stratified 80:20 split with seed 42: 2,000 training rows (1,000 per class), 500 held-out rows (250 per class). The split is recorded in `data/classification_split.csv`. Age uses a fixed reference year of 2026; price and mileage use log1p. Imputation, numeric scaling and categorical encoding live inside each model pipeline and are fitted within training folds. No resampling is needed.

Five-fold stratified **training CV macro F1** chooses hyperparameters and the final classifier. Baseline settings are eligible in each Part B search. Consequently a tuned model can retain its baseline settings, and a better CV score can still give a lower held-out score. The test partition is used for final reporting, not parameter selection. The CV scores used in selection are not nested-CV estimates of generalization.

Required classification results include accuracy, weighted precision/recall/F1, confusion matrices and ROC-AUC; macro F1, balanced accuracy, Automatic-class measures and PR curves are also provided. Automatic is the positive class for ROC/PR. All models share exactly the same training/test data.

Clustering retains the original seed-42 80:20 split (3,207 fit rows; 802 holdout rows), train-only imputation/scaling, and five-fold tuning. K-Means searches k=2–6. Hierarchical search compares k=2–6 with Ward, complete and average linkage. Validation uses native K-Means prediction or the explicitly documented hierarchical centroid proxy. Collapsed validation partitions are rejected. Silhouette, Davies–Bouldin and Calinski–Harabasz are reported separately for fitted and held-out partitions. Higher silhouette/CH and lower DB are better; no ground-truth cluster accuracy is claimed.

PCA is for visualization only and uses the same fitted projection for each model's before/after figures. t-SNE uses a fixed 1,200-row sample. The hierarchical dendrogram uses a labelled 600-row sample and is truncated to 25 leaves; it is not a drawing of all 3,207 training observations. Silhouette computations use at most 2,000 observations where specified in the code.

### Final model selection

Random Forest is the final classifier because it has the highest five-fold training CV macro F1. Its hyperparameter search selected the baseline parameter configuration, so the before- and after-tuning scores are identical; exact CV ties are reported using the CV-selected **After tuning** artifact. The held-out accuracy is 86.8% and macro F1 is approximately 0.868. Test metrics never choose the winner.

Hyperparameter tuning still produces a documented improvement: Bagging increases training CV macro F1 by approximately 0.0135, held-out macro F1 by 0.0200, and held-out ROC-AUC by approximately 0.0169. These results describe the balanced sample, not all used cars.

## Faster startup

The dashboard reads a small precomputed JSON summary and local artwork. It imports no training modules, loads no model objects, and creates no interactive plots until the selected page needs them. Models load on prediction submission and are cached; data and assets are cached by file modification time. Input forms submit together, and only the selected view is rendered. The GUI performs no fitting, hyperparameter search or source downloading.

An isolated Streamlit AppTest measurement in the preparation environment reduced first app execution from **2.612 s to 0.131 s** (about 95%). This is server-side script execution after the test framework import, not full browser/network startup or a guarantee for another machine. Full measurements and method are in `results/performance_check.json` and `VERIFICATION.md`.

## Reproduce or verify

From the project root, using the same activated Python 3.12 environment:

```bash
python tests/verify_project.py
python tests/test_gui.py
```

To rerun the existing notebook cells and regenerate fitted artifacts:

```bash
python scripts/execute_inprocess.py
```

Alternatively, `python scripts/execute_notebooks.py` uses a Jupyter kernel. Open the notebooks in VS Code with its Jupyter extension, or install JupyterLab separately. Full retraining is much slower than normal GUI use: it executes all cross-validation candidates, not just predictions. The in-process runner was used successfully for this delivery. Stop/restart the GUI after retraining so every session uses the refreshed artifacts.

`python scripts/build_notebooks.py` regenerates notebook source and clears existing notebook outputs and team edits. It is a developer utility, not a required startup step. Back up original team commentary before using it. To refresh only the summary assets after rerunning models, use `python scripts/write_summary.py` and then `python scripts/export_gui_assets.py`.

`python scripts/prepare_classification_dataset.py` independently reproduces the source sampling into `reproduced_data/`; it does not replace the supplied dataset or models. Source updates can change the result, so compare the recorded source archive and output SHA-256 values before adopting a new sample.

## Guidelines, authorship and remaining team actions

Read **GUIDELINE_CHECKLIST.md** beside the supplied original PDF. The package covers Review 2 classification and clustering. The earlier regression track was not supplied and is not recreated here. The guideline expects dataset continuity across reviews; the requested change to UK listings is documented and Part A is rerun, but your team must establish instructor acceptance of that change.

AI assistance was used for code scaffolding, preprocessing implementation, experiment execution, validation and interface design. Results are computed from actual runs; they are not fabricated improvements. The guideline requires original team analysis: complete the notebook interpretation prompts yourselves, explain why tuning helps or hurts, discuss cluster meaning and limitations, record member responsibilities, and understand the code for the viva. Do not submit AI-generated commentary as your own analysis.

The ZIP does not create meaningful external GitHub commit history or a public deployment. Maintain genuine incremental commits in your team repository. For the optional deployment bonus, deploy the complete project with entry point `app/app.py`, Python 3.12 and pinned requirements on a suitable Streamlit host; include the supplied artifacts so server startup does not train models. Public deployment has not been performed or tested here.

See **VERIFICATION.md** for the checks that passed and their boundaries. The native Windows/macOS launchers have been reviewed but were not executed on those operating systems.

<!-- GENERATED_RESULTS -->
## Executed result summary

These tables are generated directly from the executed notebooks. Test deltas are after minus before;
they are not a promise that tuning improves every model. Fold-selected settings may match the baseline.

### Part B classification

| Model | Macro F1 before | Macro F1 after | Test delta | CV macro F1 before | CV macro F1 after |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8679 | 0.8679 | 0.0000 | 0.8550 | 0.8550 |
| AdaBoost | 0.8138 | 0.8049 | -0.0089 | 0.8052 | 0.8099 |
| Gradient Boosting | 0.8577 | 0.8577 | 0.0001 | 0.8319 | 0.8373 |
| Bagging | 0.8420 | 0.8620 | 0.0200 | 0.8354 | 0.8489 |
| MLP | 0.8459 | 0.8459 | 0.0000 | 0.8449 | 0.8449 |

### Clustering

Training partition metrics are below. Silhouette and Calinski–Harabasz are higher-is-better;
Davies–Bouldin is lower-is-better. See the full CSV for separately labelled holdout and CV results.

| Model | Stage | k | Silhouette | Davies_Bouldin | Calinski_Harabasz |
| --- | --- | --- | --- | --- | --- |
| K-Means | Before tuning | 3 | 0.3730 | 0.8993 | 2992.6547 |
| K-Means | After tuning | 2 | 0.4234 | 0.8806 | 3120.8159 |
| Agglomerative | Before tuning | 3 | 0.3595 | 0.8799 | 2622.7525 |
| Agglomerative | After tuning | 2 | 0.3951 | 0.8755 | 2161.4716 |

The complete ten-model comparison is in `results/ten_model_comparison.csv` and notebook 13.
These computed results still need your team's own explanation and discussion.
