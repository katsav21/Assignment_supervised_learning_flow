# Supervised Learning Flow — Breast Cancer Diagnosis (WDBC)

> **TL;DR:** Best pipeline = **StandardScaler + LogisticRegression (liblinear, C=10)**.  
> **5-Fold CV F1 (Malignant): 0.9838 · Test F1 (Malignant): 0.9677**

<p align="center">
  <em>End-to-end ML pipeline: EDA → Feature Engineering → GridSearchCV (Stratified 5-Fold) → Final Training → Test Evaluation</em>
</p>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Visualizations](#visualizations)
- [Reproducibility](#reproducibility)
- [Getting Started](#getting-started)
- [Assignment Compliance](#assignment-compliance)
- [Authors](#authors)
- [License](#license)

---

## Project Overview
This repository implements a complete supervised learning pipeline to **classify breast tumors** as **Malignant (1)** or **Benign (0)** using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The pipeline follows best practices in experimental design and validation.

- **Course:** Machine Learning — Supervised Learning Flow Assignment  
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), **569** samples, **30** features (FNA descriptors)

## Problem Statement
**Objective:** Build a reliable binary classifier that predicts whether a breast mass is **Malignant (1)** or **Benign (0)** from 30 quantitative FNA descriptors.

**Clinical Relevance:** Missing malignant cases (FN) can delay treatment. We therefore evaluate with **F1 on the malignant class**, balancing **precision** and **recall**.

## Dataset Description
- **Total Samples:** 569  
- **Features:** 30 = 10 base properties × 3 variants (**mean**, **SE**, **worst**)  
- **Target:** `Malignant=1` (positive) vs `Benign=0`  
- **Missing Values:** None (canonical dataset)  
- **Class Balance:** ~63% Benign / ~37% Malignant (mild imbalance)  
- **Train/Test:** Provided as `cancer_train.csv` and `cancer_test.csv` (no re-split)

**Base properties:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

## Project Structure
```
.
├── Assignment_supervised_learning_flow.ipynb   # Main notebook
├── assignment_ml_flow_instructions_short.pdf       # Assignment instructions
├── README.md                                       # This file
└── data/
    ├── cancer_train.csv                            # Training data
    ├── cancer_test.csv                             # Test data
    └── breast_cancer_description.txt               # Dataset notes
```

## Methodology
1. **Data Loading & EDA** — sanity checks, class balance, feature distributions, scatter plots, correlation heatmap.  
2. **Experiments (Grid Search)** — `Pipeline(scaler → selector → classifier)` + **Stratified 5-Fold**, scoring **F1 (Malignant)**.  
   - **Feature engineering:** none / StandardScaler / MinMaxScaler; optional `SelectKBest` (ANOVA F / Mutual Information).  
   - **Models & hypers:** Logistic Regression (`C`, `class_weight`), Random Forest (`n_estimators`, `max_depth`), SVM-RBF (`C`, `gamma`, `class_weight`).  
3. **Model Selection** — pick the highest **mean CV F1**.  
4. **Final Training** — retrain the winning pipeline on the **entire training set**.  
5. **Test Evaluation** — predict on test; report **F1 (Malignant)**, classification report, confusion matrices; PR/ROC; first 5 predictions.
6. **Reproducibility** — seed=42, environment printouts, saved artifacts.

## Results
### Best Model Configuration
- **Pipeline:** **StandardScaler / LogisticRegression**
- **Hyperparameters:** solver=`liblinear`, **C=10**, `class_weight=None`
- **Selector:** `passthrough` (feature selection not required for the winner)

### Performance
| Metric | Cross-Validation (5-Fold) | Test Set |
|---|---:|---:|
| **F1 (Malignant)** | **0.9838** | **0.9677** |

**Takeaway:** Very small CV→test gap → **strong generalization**. If minimizing false negatives is critical, tune **class weights** or the **decision threshold**, and calibrate probabilities.

## Visualizations
Included in the notebook:
- **Class Distribution** — justifies F1(pos=1) + stratified CV.  
- **Histograms/Boxplots** — malignant values skew higher on key features → separation.  
- **2D Scatter** — linear margin works well when scaled; non-linear models also tested.  
- **Correlation Heatmap** — geometric feature blocks (radius–perimeter–area) → regularization/selection.  
- **PR & ROC Curves** — threshold-agnostic diagnostics; PR aligns with F1 focus.  
- **Confusion Matrices** — FP/FN trade-offs (counts & normalized).  
- **Score Distributions** — decide whether 0.5 threshold is optimal.

## Reproducibility
- **Seeds:** `random_state=42` (CV & models)  
- **Artifacts (optional, saved by notebook):**  
  - `best_model_breast_cancer.pkl`  
  - `feature_columns.json`  
  - `grid_results_full.csv`  
  - `test_predictions.csv`  
- **Environment:** Python & library versions printed in the notebook; `sklearn.show_versions()` included.

## Getting Started
```bash
# Create & activate an environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install basics
pip install -U pip
pip install scikit-learn pandas numpy matplotlib joblib

# Run the notebook
jupyter notebook "Assignment_supervised_learning_flow (3).ipynb"
# In Part 3: set RUN_GRID_SEARCH=True, run the grid
# In Part 4: retrain best estimator on full train
# In Part 5: evaluate on test and save artifacts
```

## Assignment Compliance
- ✅ **Part 1:** Student details, AI prompts, dataset paragraph  
- ✅ **Part 2:** Load train/test; show `.head()`; ≥3 EDA visualizations with explanations  
- ✅ **Part 3:** GridSearchCV (Stratified 5-Fold), F1(pos=1), full permutations table, best config  
- ✅ **Part 4:** Retrain best pipeline on the full training set  
- ✅ **Part 5:** Predict full test set; first 5 predictions; F1(pos=1); diagnostics & plots

## Authors
- _FirstName LastInitial (####)_, _FirstName LastInitial (####)_

## License
Academic use only — part of a course assignment.
