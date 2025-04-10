from sklearn.pipeline import Pipeline # For chaining preprocessing and modeling steps together
from sksurv.ensemble import RandomSurvivalForest # Survival model that extends random forests to handle censored data
from sklearn.model_selection import train_test_split # For splitting data into training and testing subsets
from sksurv.metrics import concordance_index_censored # For evaluating survival models using concordance index (C-index)

from b_data_preprocessing import load_raw_data, get_preprocessor # Load utility functions for data and preprocessing pipeline

from lifelines import CoxPHFitter # Cox Proportional Hazards model from lifelines library
from lifelines.utils import concordance_index # Standard concordance index function from lifelines
import pandas as pd # For data manipulation

# -------------------------------------------------------------
# Function: train_rsf_pipeline
# Description:
# Trains a Random Survival Forest (RSF) model using a full
# pipeline that includes preprocessing and modeling.
# Returns the trained model, test data, and predictions.
# -------------------------------------------------------------
def train_rsf_pipeline():
    # Load raw features and survival target
    X, y = load_raw_data() # Load cleaned features and survival outcome using helper function

    # Load preprocessing pipeline
    preprocessor = get_preprocessor() # Load preprocessing steps (scaling, encoding)

    # Define full pipeline (preprocessor + model)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # First apply preprocessing
        ('model', RandomSurvivalForest(n_estimators=200, random_state=42)) # Then train RSF model with 200 trees, fixed seed
    ]) # Pipeline ensures that preprocessing and model training are applied consistently to both train and test data

    # Split data (Train-test split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25) # Split into 80/20 training/testing sets

    # Fit model (train the pipeline on training data)
    pipeline.fit(X_train, y_train) # Train the entire pipeline (including both preprocessing and RSF model)

    # Predict survival outcomes on the test set
    y_pred = pipeline.predict(X_test) # Predict survival function values for test set (typically risk or survival time estimation)

    # Evaluate using concordance index (C-index)
    c_index = concordance_index_censored(y_test["Status OS"], y_test["OS at baseline"], y_pred)[0] # index [0] retrieves only the concordance index value
    # Calculate concordance index — measures how well predicted rankings match actual outcomes
    # Only the first element of the tuple is needed (the concordance score), the full tuple gives + performance stats

    print(f"RSF model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}") # Show model performance. format c_index to 4 dp

    return pipeline, X_test, y_test, y_pred # Return everything needed for downstream evaluation


# -------------------------------------------------------------
# Function: train_coxph_model
# Description:
# Trains a Cox Proportional Hazards model using the lifelines
# package, after preprocessing and combining features with survival
# targets. Returns model performance and sample count.
# -------------------------------------------------------------
def train_coxph_model():
    # Load features and target
    X, y = load_raw_data() # Load original features and target
    preprocessor = get_preprocessor() # Load the same preprocessing pipeline

    # Apply preprocessing to features
    X_preprocessed = preprocessor.fit_transform(X) # Apply transformation to the full dataset (no leakage, since train_test_split is after this)

    # Ensure result is a DataFrame for lifelines compatibility
    if not isinstance(X_preprocessed, pd.DataFrame):
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())
        # Convert output to DataFrame so that lifelines (which expects DataFrame input) works correctly

    # Combine into one DataFrame for lifelines (combining preprocessed features and survival labels)
    df = X_preprocessed.copy() # Make a copy of the preprocessed features
    df["OS at baseline"] = y["OS at baseline"] # Add time-to-event column
    df["Status OS"] = y["Status OS"] # Add event status (1=event occurred, 0=censored)

    # Split into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=25) # Split dataset for evaluation

    # Fit CoxPH model
    cph = CoxPHFitter(penalizer=0.1) # Instantiate Cox model with L2 penalty (regularization helps stability)
    cph.fit(train_df, duration_col="OS at baseline", event_col="Status OS") # Fit model on training set using lifelines API

    # Predict partial hazards on test set
    test_predictions = cph.predict_partial_hazard(test_df) # Predict partial hazard (log-risk score) for test set

    # Evaluate model using concordance index
    c_index = concordance_index(test_df["OS at baseline"], -test_predictions, test_df["Status OS"])
    # Evaluate performance using concordance index; note: use -test_predictions because higher hazard = worse survival

    print("Cox PH model trained successfully.")
    print(f"Concordance Index (Test Set): {c_index:.4f}") # Display C-index

    return c_index, len(test_df) # Return performance score and number of test samples
