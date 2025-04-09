# Breast Cancer Survival Prediction using Machine Learning

This project applies survival analysis and machine learning techniques to predict breast cancer patient survival times using clinical features and ctDNA mutation data. The pipeline compares the performance of a **Random Survival Forest** model with a traditional **Cox Proportional Hazards** model, using the concordance index for evaluation.

---

## Project Objectives

- Use real-world clinical and genetic mutation data (ctDNA) to model patient survival outcomes.
- Build and evaluate a machine learning pipeline using Random Survival Forests (RSF).
- Benchmark RSF performance against the traditional Cox Proportional Hazards (Cox PH) model.
- Visualize Kaplan-Meier survival curves across clinically relevant subgroups.
- Structure code following reproducible, modular, and professional standards.

---

## Methods Overview

- **Exploratory Data Analysis (EDA)**: Kaplan-Meier survival curves by mutation, metastatic site, and subtype.
- **Preprocessing**: Imputation, one-hot encoding, feature scaling, survival target formatting.
- **Modeling**: 
  - `RandomSurvivalForest` from `scikit-survival`
  - `CoxPHFitter` from `lifelines`
- **Evaluation**: Concordance index (C-index) to assess model performance.

---

## Project Structure

```
breast_cancer_survival_project/
├── data/
│   ├── breast_cancer_data_raw.csv
├── scripts/
│   ├── a_exploratory_analysis.py
│   ├── b_data_preprocessing
│   ├── c_survival_models.py
│   └── d_evaluate_models.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── figures/
│   └── figures1-7
├── results/
│   └── rsf_evaluation_results.csv
└── README.md
```


