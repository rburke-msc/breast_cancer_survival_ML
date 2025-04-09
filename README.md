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

breast_cancer_survival_project/ ├── data/ │ ├── raw/ # Raw input data │ └── processed/ # Cleaned & preprocessed data │ ├── scripts/ │ ├── data_preprocessing.py # Data loading and preprocessing functions │ ├── exploratory_analysis.py # Kaplan-Meier plots and visualizations │ ├── survival_models.py # Model training (RSF and Cox PH) │ └── evaluate_models.py # Evaluation + performance summary │ ├── figures/ # Auto-generated EDA plots ├── results/ # Concordance index CSVs ├── notebooks/ # (Optional) Interactive Jupyter notebooks └── README.md # Project overview


Structure is as follows:

breast_cancer_survival_project/
├── data/
│   ├── raw/
│   │   └── breast_cancer_data_raw.csv
│   └── processed/
│       └── breast_cancer_data_processed.csv
├── scripts/
│   ├── data_preprocessing.py
│   ├── exploratory_analysis.py
│   ├── survival_models.py
│   └── evaluate_models.py
├── notebooks/
│   └── survival_analysis_pipeline.ipynb
├── figures/
│   └── (your images here)
├── results/
│   └── survival_predictions.csv
└── README.md

