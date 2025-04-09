from sklearn.pipeline import Pipeline
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from data_preprocessing import load_raw_data, get_preprocessor

def train_rsf_pipeline():
    # Load raw features and survival target
    X, y = load_raw_data()

    # Get the preprocessing pipeline
    preprocessor = get_preprocessor()

    # Define full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomSurvivalForest(n_estimators=200, random_state=42))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    # Fit model
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    c_index = concordance_index_censored(y_test["Status OS"], y_test["OS at baseline"], y_pred)[0]

    print(f"RSF model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}")

    return pipeline, X_test, y_test, y_pred


from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd

def train_coxph_model():
    # Load and preprocess data
    X, y = load_raw_data()
    preprocessor = get_preprocessor()

    # Preprocess features only (output should be a DataFrame)
    X_preprocessed = preprocessor.fit_transform(X)
    if not isinstance(X_preprocessed, pd.DataFrame):
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())

    # Combine into one DataFrame for lifelines
    df = X_preprocessed.copy()
    df["OS at baseline"] = y["OS at baseline"]
    df["Status OS"] = y["Status OS"]

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=25)

    # Fit CoxPH model
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col="OS at baseline", event_col="Status OS")

    # Predict partial hazards
    test_predictions = cph.predict_partial_hazard(test_df)

    # Evaluate
    c_index = concordance_index(test_df["OS at baseline"], -test_predictions, test_df["Status OS"])

    print("Cox PH model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}")

    return c_index, len(test_df)
