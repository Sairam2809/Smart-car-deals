# TORQUEX verification report

Core project and GUI checks were rerun on 21 October 2026 in the project Windows virtual environment with the pinned dependencies. Earlier browser and performance checks were completed in the preparation environment as described below. These are checks actually performed, not a guarantee that every operating-system or data scenario is covered.

## Executed experiments

- All **13 notebooks / 91 code cells** executed successfully. Outputs are preserved in the delivered notebooks.
- Notebook schemas validate; execution counts are ordered; no error outputs; **90 embedded plots**.
- Five Part B baseline/tuned experiments, five Part A baseline reruns, K-Means and hierarchical baseline/tuned experiments produced **19 saved model artifacts**.
- Actual measured scores populate the comparisons. Tuning is not forced to improve held-out results; unchanged and negative deltas remain visible.
- Execution used `scripts/execute_inprocess.py`, which runs real notebook code cells and captures their outputs. The alternative socket-based Jupyter runner is supplied but was not used for the delivered execution.

## Dataset and model checks — PASS

`python tests/verify_project.py`

- 2,500 classification records; exactly 1,250 Manual and 1,250 Automatic; zero missing values, exact duplicate rows or feature-identical duplicate rows.
- Stratified 2,000/500 split; 1,000/250 observations per class; disjoint row membership and no duplicated feature records across partitions.
- Saved classifier predictions reproduce exported metrics for all **15 variants**.
- Cross-validation fold identities match; preprocessing is inside the fitted pipelines; no resampling stage is present.
- Classification handles unseen categorical values and missing predictor values through the fitted encoder/imputer.
- Final model selection follows training CV scores, not held-out scores; exact ties prefer the CV-selected after-tuning artifact.
- Original clustering data hash matches the preceding package byte for byte.
- Clustering partitions are disjoint; saved labels, all before/after metrics, holdout assignments, five-fold searches and cluster profile totals are consistent for all four variants.
- CSV prediction validation covers correct schema, missing columns, empty or oversized datasets, nonnumeric/infinite values, invalid ranges, and batch/single prediction agreement.
- Updated classification source, split metadata, all-after-tuning table and car artwork are present.

See `verification.log` for the recorded test output. The source-row membership audit was performed when preparing the real-record sample; `data/classification_provenance.csv` and `data/classification_source_audit.json` retain its evidence. The project verification script checks the supplied data, not a fresh download of the upstream source.

## GUI interaction checks — PASS

`python tests/test_gui.py` exercises the actual Streamlit application through its AppTest interface:

- Dashboard and classification prediction route.
- Prediction submission with each of the ten classifiers.
- Five Part B diagnostic views; before/after, all-after-tuning and consolidated ten-model comparisons.
- K-Means and hierarchical new-car assignment.
- Empty cluster filter handling and both clustering comparison views.
- Complete evaluation-metrics page; the removed Data Garage route is absent from navigation.

The final GUI test run passed after synchronising the checks with the current navigation and removing the obsolete review-guide and Data Garage routes.

## Real browser checks — PASS

A local Streamlit server was opened in headless Chromium (Chrome Headless Shell 131, Playwright 1.63). The check:

- Opened and visually inspected the dashboard, single-car form/result, model comparison and cluster explorer at 1440 × 1040.
- Opened and visually inspected the narrow dashboard at 390 × 844; navigation collapses and cards stack.
- Submitted the single-car prediction form.
- Uploaded the three-row CSV template, ran batch inference, downloaded the result and verified all three prediction rows and labels.
- Uploaded an invalid-schema CSV and confirmed an explicit missing-columns error.
- Captured no browser page-error events in this run.

The batch workflow uses a submit form so an upload change and prediction are processed together. CSV downloads do not rerun the app. Internal QA screenshots are not distributed with this package. This browser check does not constitute an exhaustive device/accessibility audit or public-hosting test. Machine-readable results are in `results/browser_check.json`.

## Performance evidence

`results/performance_check.json` contains the original and TORQUEX measurements. Each was measured in an isolated process with Streamlit AppTest, after importing the test framework, with one first execution and three warm reruns.

| Measurement | Previous app | TORQUEX |
|---|---:|---:|
| First server-side app execution | 2.612 s | 0.131 s |
| Warm reruns, observed range | 0.052–0.057 s | 0.035–0.041 s |
| Training modules / sklearn / matplotlib / Plotly loaded by landing route | Yes | No |

First app execution was approximately **95% shorter** in this measurement. It excludes Python process startup, installation, browser loading and network delays. It is not a statistical benchmark or a guarantee for the user's computer. A local browser also reached the rendered dashboard successfully, but its timing is not used to claim an end-to-end speedup. Recheck your environment with `python tests/benchmark_startup.py`.

## Packaging and boundaries

- Python source compiled successfully; POSIX launcher shell syntax checked.
- The ZIP was checked for integrity and includes the complete directory structure, data, executed notebooks, model artifacts, results, original guidelines and documentation.
- No virtual environment, downloaded browser, source archive, transient cache or internal screenshots are included.
- Native Windows and macOS launchers were reviewed but not run on those operating systems. Use Python 3.12 and pinned dependencies.
- Public deployment and external GitHub history were not created. The earlier regression track was not supplied. Team interpretations, member contributions and academic acceptance of the new classification dataset remain the team's responsibility, as recorded in the guideline checklist.
