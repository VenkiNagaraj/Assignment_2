# Water Quality Classification – BITS WILP AI/ML SEM 1, ML Assignment 2

Predicting whether a water sample is **safe to drink** with five classification models, compared on six evaluation metrics and served through an interactive Streamlit app.

- **GitHub repository:** https://github.com/VenkiNagaraj/Assignment_2
- **Live Streamlit app:** _add the link here after deployment_

---

## a. Problem statement

Given the concentrations of 20 substances in a water sample (metals, chemicals, bacteria, viruses and radioactive elements), predict whether the water is **safe to drink** (`is_safe = 1`) or **not safe** (`is_safe = 0`).

This is a **binary classification** problem on an **imbalanced** dataset: only about 11% of the samples are safe. Many real-world datasets are imbalanced, so a key part of this assignment is handling the imbalance properly — choosing metrics beyond accuracy and comparing models with and without class weighting.

## b. Dataset description

| Property | Value |
|---|---|
| Source | [Water Quality – Kaggle (uploaded by MsSmartyPants)](https://www.kaggle.com/datasets/mssmartypants/water-quality) |
| Raw file | `dataset/waterQuality1.csv` |
| Instances | 7,999 raw → **7,996** after cleaning |
| Features | **20**, all numeric |
| Target | `is_safe` — 0 = not safe, 1 = safe |
| Class balance | 7,084 not safe (88.6%) vs **912 safe (11.4%)** |

**Features** (concentration of each substance; the dataset description gives a "dangerous if greater than" limit for each):

| Group | Features |
|---|---|
| Metals | aluminium, arsenic, barium, cadmium, chromium, copper, lead, mercury, selenium, silver |
| Other chemicals | ammonia, chloramine, flouride, nitrates, nitrites, perchlorate |
| Microbes | bacteria, viruses |
| Radioactive | radium, uranium |

**Observations about the data**

- **Synthetic data.** The dataset description says it was created from *imaginary* data of water quality in an urban environment, for educational use. That is fine here — the purpose is learning to train and evaluate models — but the results should not be read as real water-science findings.
- **Data cleaning.** 3 rows contained the Excel error value `#NUM!` in both `ammonia` and `is_safe`. Since the target itself was corrupted, these rows were dropped. There were no other missing values and no duplicate rows. (`ammonia` also has a minimum of −0.08, a physically impossible value that was left as is.)
- **Very different feature scales** — from mercury (max 0.01) to perchlorate (max 60). Features are therefore standardised (`StandardScaler`) before training, which matters for Logistic Regression, KNN and SVM.
- **The published thresholds do not determine the label.** Only 1 of the 912 "safe" samples is below all 20 limits, and 91% of "safe" samples exceed the lead limit. The labels follow a hidden, non-linear rule, so the models have to learn it from the data — simple threshold rules would not work.

## c. GitHub repository link

https://github.com/VenkiNagaraj/Assignment_2

```
Assignment_2/
├── app.py                  # Streamlit app (loads saved models, no retraining)
├── model_training.ipynb    # Data cleaning, training, tuning and evaluation of all 5 models
├── requirements.txt        # Pinned dependencies
├── README.md
├── test_data.csv           # Held-out test set (1,600 rows, 20 features + is_safe)
├── dataset/
│   └── waterQuality1.csv   # Raw dataset from Kaggle
└── model/                  # Saved pipelines (StandardScaler + classifier)
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## d. Models used

### Approach

1. **Stratified 80/20 train/test split** (6,396 train / 1,600 test). Stratification keeps the 11.4% safe share identical in both sets. The test set is saved as `test_data.csv` and keeps its natural imbalance, so the metrics reflect real-world conditions.
2. **Pipeline = `StandardScaler` → classifier.** The scaler is fitted on training data only, so no information leaks from the test set. Each saved model file contains its own scaler, so the app can use raw CSV values directly.
3. **Hyperparameter tuning with `GridSearchCV`** — 5-fold stratified cross-validation on the training set, scored on **F1** (accuracy is misleading here: always predicting "not safe" already scores ~89%).
4. **Imbalance handling:** `class_weight` was tuned as `[None, "balanced"]` for the models that support it (Logistic Regression, Decision Tree, SVM). KNN and Naive Bayes have no such option.
5. **Overfitting check:** train and test metrics are compared for every model in the notebook.

| Model | Grid searched | Best parameters |
|---|---|---|
| Logistic Regression | C ∈ {0.01, 0.1, 1, 10, 100} × class_weight | C = 0.01, class_weight = balanced |
| Decision Tree | max_depth ∈ {3, 5, 8, 12, None} × min_samples_leaf ∈ {1, 5, 10, 20} × class_weight | max_depth = 12, min_samples_leaf = 1, class_weight = balanced |
| KNN | n_neighbors ∈ {3, 5, 7, 11, 15, 21} × weights ∈ {uniform, distance} | n_neighbors = 7, weights = uniform |
| Naive Bayes (Gaussian) | var_smoothing ∈ {1e-9 … 1e-1} | var_smoothing = 1e-3 |
| SVM (RBF kernel) | C ∈ {0.1, 1, 10} × gamma ∈ {scale, 0.01, 0.1} × class_weight | C = 10, gamma = scale, class_weight = None |

### Comparison table (test set, 1,600 samples)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8006 | 0.8407 | 0.3283 | 0.7198 | 0.4509 | 0.3895 |
| Decision Tree | **0.9588** | 0.9262 | 0.7900 | **0.8681** | **0.8272** | **0.8050** |
| KNN | 0.9194 | 0.8728 | 0.7912 | 0.3956 | 0.5275 | 0.5240 |
| Naive Bayes | 0.8481 | 0.7930 | 0.3875 | 0.5769 | 0.4636 | 0.3893 |
| Support Vector Machine | 0.9425 | **0.9539** | **0.7961** | 0.6648 | 0.7246 | 0.6962 |

Precision, Recall and F1 are for the **safe (1)** class. The test set has 182 safe and 1,418 not-safe samples.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Weakest model together with Naive Bayes (F1 0.45). It can only draw a straight-line decision boundary, but the labels follow a non-linear rule, so it **underfits** — train and test scores are close (F1 0.48 vs 0.45), so it is not overfitting. Class weighting helped clearly: `balanced` raised cross-validated F1 from 0.35–0.45 to ~0.48 at every C, while C itself barely mattered. Recall is decent (0.72) but precision is only 0.33: it labelled **268 unsafe samples as safe**, the most dangerous kind of error for drinking water. |
| Decision Tree | **Best model overall** (F1 0.83, MCC 0.81, recall 0.87). Trees split on one feature at a time and can model the non-linear, threshold-like rule behind the labels, which linear models cannot. Cadmium (33% of importance), aluminium and chloramine were the most used features. Without a depth limit the tree reached train F1 = 1.00 (memorising the training data); cross-validation picked max_depth = 12, which still overfits somewhat (train F1 0.94 vs test 0.83) but generalises best. A shallower tree (depth 8, min_samples_leaf 20) scored almost as well with a much smaller train/validation gap. |
| KNN | High precision (0.79) but very low recall (0.40): it found only **72 of 182** safe samples. KNN has no class-weight option, so in most neighbourhoods the majority "not safe" samples out-vote the rare safe ones. Accuracy (0.92) looks good only because of the imbalance — F1 and MCC expose the weakness. `weights = distance` reached train F1 = 1.00 (memorisation), and cross-validation correctly preferred uniform voting with k = 7. |
| Naive Bayes | Similar to Logistic Regression (F1 0.46, MCC 0.39) and the lowest AUC (0.79). Gaussian NB assumes each feature is an independent bell curve within each class; many features here are strongly skewed (e.g. aluminium has median 0.07 but max 5.05), so the assumption does not hold. No overfitting (train ≈ test), and tuning `var_smoothing` made almost no difference. |
| Support Vector Machine | Second best (F1 0.72, MCC 0.70) and the **highest AUC (0.954)** — it ranks safe vs unsafe samples best of all models, and has the highest precision (0.80, only 31 false "safe" calls). At the default decision threshold it misses a third of the safe samples (recall 0.66). It overfits noticeably (train F1 0.92 vs test 0.72). Unlike LR and DT, `class_weight = balanced` did not help. The best C was the largest value in the grid, so a larger C might improve it further. |
| **Overall winner for this dataset** | **Decision Tree** — best Accuracy, Recall, F1 and MCC, and the fewest missed safe samples (24). The non-linear, rule-like structure of the labels suits a tree. SVM is the runner-up and would be the choice if ranking quality (AUC) or fewest false "safe" predictions mattered most. |

---

## Streamlit app

The app (`app.py`) loads the five saved models and evaluates them on uploaded test data **without retraining**.

- **Dataset upload (CSV)** — upload a test CSV, or use the bundled `test_data.csv` (also downloadable from the sidebar).
- **Model selection dropdown** — choose any of the five classifiers.
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC.
- **Confusion matrix and classification report** for the selected model.
- **Compare all models** tab — all five models side by side on the same data.

The uploaded CSV must contain the 20 feature columns and the `is_safe` column. Rows with non-numeric values (e.g. `#NUM!`) are dropped with a warning.

### Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

To retrain the models, run `model_training.ipynb` top to bottom (it rewrites `model/` and `test_data.csv`).
