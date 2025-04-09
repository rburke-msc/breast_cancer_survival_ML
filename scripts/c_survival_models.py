from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from b_data_preprocessing import load_raw_data, get_preprocessor

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

# -------------------------------------------------------------
# Function: train_rsf_pipeline
# Description:
# Trains a Random Survival Forest (RSF) model using a full
# pipeline that includes preprocessing and modeling.
# Returns the trained model, test data, and predictions.
# -------------------------------------------------------------
def train_rsf_pipeline():
    # Load raw features and survival target
    X, y = load_raw_data()

    # Load preprocessing pipeline
    preprocessor = get_preprocessor()

    # Define full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomSurvivalForest(n_estimators=200, random_state=42))
    ])

    # Split data (Train-test split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Fit model (train the pipeline on training data)
    pipeline.fit(X_train, y_train)

    # Predict survival outcomes on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate using concordance index (C-index)
    c_index = concordance_index_censored(y_test["Status OS"], y_test["OS at baseline"], y_pred)[0]

    print(f"RSF model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}")

    return pipeline, X_test, y_test, y_pred


# -------------------------------------------------------------
# Function: train_coxph_model
# Description:
# Trains a Cox Proportional Hazards model using the lifelines
# package, after preprocessing and combining features with survival
# targets. Returns model performance and sample count.
# -------------------------------------------------------------
def train_coxph_model():
    # Load features and target
    X, y = load_raw_data()
    preprocessor = get_preprocessor()

    # Apply preprocessing to features
    X_preprocessed = preprocessor.fit_transform(X)

    # Ensure result is a DataFrame for lifelines compatibility
    if not isinstance(X_preprocessed, pd.DataFrame):
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())

    # Combine into one DataFrame for lifelines (combining preprocessed features and survival labels)
    df = X_preprocessed.copy()
    df["OS at baseline"] = y["OS at baseline"]
    df["Status OS"] = y["Status OS"]

    # Split into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=25)

    # Fit CoxPH model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col="OS at baseline", event_col="Status OS")

    # Predict partial hazards on test set
    test_predictions = cph.predict_partial_hazard(test_df)

    # Evaluate model using concordance index
    c_index = concordance_index(test_df["OS at baseline"], -test_predictions, test_df["Status OS"])

    print("Cox PH model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}")

    return c_index, len(test_df)
