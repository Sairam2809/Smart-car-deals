# TORQUEX: Review 2 guideline checklist

Reference: `data/capstone_guidelines.pdf`, especially sections 3.2, 3.3, 4, 6 and 7.

| Requirement | Location / status |
|---|---|
| Five Part B classifiers | Separate notebooks 06–10: Random Forest, AdaBoost, Gradient Boosting, Bagging with Decision Tree, MLP |
| Same data across Part A and B | All ten are rerun on the new 2,500-row UK dataset and the same split. This changes the original Review 1 dataset; the instructor must accept that change. |
| Final ten-model comparison | Notebook 13 and `results/ten_model_comparison.csv`; Part A baseline reruns and each Part B CV-selected version |
| Before/after tuning per model | Notebook 06–10, `results/part_b_before_after.csv`, GUI Model comparisons |
| After-tuning comparison | `results/after_tuning_comparison.csv`, notebook 13 and dedicated GUI view |
| Improvement reported | Actual deltas in `results/tuning_deltas.csv`; no claim that every model improves |
| Classification metrics | Accuracy, weighted precision/recall/F1, confusion matrices, ROC-AUC; extra macro F1, balanced accuracy, minority/class metrics and PR plots |
| Final selection and justification | Five-fold training macro F1 rule, exported selection; team adds substantive interpretation |
| K-Means and hierarchical clustering | Notebooks 11 and 12; original `used_cars.csv` retained |
| Clustering metrics | Silhouette, Davies–Bouldin, Calinski–Harabasz before/after; separate CV/holdout metrics |
| Elbow and dendrogram | K-Means elbow plus CV selection; hierarchical dendrogram with explicit sample/truncation labels |
| PCA for each clustering model | Same PCA axes before/after for each method; visualization only |
| t-SNE | Included for both methods (the rubric asks for at least one) |
| Cross-validation | Five-fold classification CV for all ten; five-fold clustering validation using explicit assignment rules |
| Train-only fitting | Imputer/scaler/encoder fitted within each training fold; holdout not used for tuning |
| Clean, executed notebooks | Eight individually executed notebooks with outputs retained; readable shared functions in src/ |
| EDA and cleaning | Shapes, types, missing values, duplicate checks, distributions, correlations, outlier checks, scatterplots |
| Feature engineering | Fixed-year age and log price/distance; code is deterministic and documented |
| Markdown observations | Prompts are provided; the team must write original observations under section 7.5 |
| README and pinned environment | README, requirements.txt, quick-start scripts, source and results summaries |
| Working GUI | Local TORQUEX Streamlit app, single/batch predictions, all comparisons and cluster exploration |
| Public deployment bonus | Not deployed by this package; optional steps in README |
| GitHub meaningful commit history | Team action. The ZIP does not establish external repository history. |
| Presentation/viva and member roles | Team action: each member owns/explains a track and must understand all submitted code. |
| AI attribution | README discloses code scaffolding assistance; original team analysis remains required. |

Scope is Review 2. The earlier regression notebook was not supplied and is not recreated here.
Separate model notebooks follow the user's requested organization; notebook 13 consolidates comparisons.
