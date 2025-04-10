import pandas as pd # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For scaling numerical data and encoding categorical variables
from sklearn.compose import ColumnTransformer # For applying different preprocessing to different columns
from sklearn.pipeline import Pipeline # For chaining preprocessing and modeling steps
from sksurv.util import Surv # For converting survival labels to structured array format used by survival models

# ----------------------------------------------------
# Function: load_raw_data
# Description:
# Loads and preprocesses the raw breast cancer dataset.
# Handles tumor grade cleaning, feature engineering,
# and returns structured features and survival targets.
# ----------------------------------------------------
def load_raw_data(path="../data/raw/breast_cancer_data_raw.csv"): # Define function to load and clean data; default file path
    # Load dataset from CSV
    df = pd.read_csv(path) # Read CSV file into a DataFrame

    # ---- Clean and standardize Tumor Grade ----
    # Convert '1 or 2' into a numerical average (1.5)
    df["Tumor grade"] = df["Tumor grade"].replace('1 or 2', '1.5') # Replace ambiguous string '1 or 2' with numeric average 1.5
    df["Tumor grade"] = df["Tumor grade"].astype(float)  # Ensure the 'Tumor grade' column is float type for modeling


    # ---- Feature Engineering: Metastatic Sites ----
    # Map textual labels to binary numeric values
    df['N of met. Sites numeric'] = df['N of met. Sites'].map({'< 3': 0, '>=3': 1}) # Create binary encoding of metastasis site count


    # ---- Define categorical variables for encoding ----
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade'] # Categorical columns for encoding

    # Drop rows with missing categorical values
    df = df.dropna(subset=categorical_features) # Drop rows where any of the categorical columns are missing (ensures clean one-hot encoding)

    # ---- Define the model features and target ----
    feature_cols = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline'] + categorical_features # Final feature list
    features = df[feature_cols] # Select relevant columns as features

    # Survival target needs to be structured for survival analysis
    target = Surv.from_dataframe("Status OS", "OS at baseline", df) # Convert to structured array for survival analysis: (event, time)

    return features, target # Return feature matrix and structured survival labels


# ----------------------------------------------------
# Function: get_preprocessor
# Description:
# Returns a ColumnTransformer that handles preprocessing:
# - Scales numeric features
# - One-hot encodes categorical features
# Used in ML pipeline to standardize inputs before modeling.
# ----------------------------------------------------
def get_preprocessor():
    # Define numeric and categorical features again here
    numeric_features = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline'] # Numeric columns to be scaled
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade'] # Categorical columns to be encoded


    # ---- Numeric Transformer ----
    # StandardScaler will normalize values to mean 0 and variance 1
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()) # Pipeline that applies standard scaling (mean=0, std=1)
    ])

    # ---- Categorical Transformer ----
    # OneHotEncoder will create binary columns for each category
    categorical_transformer = OneHotEncoder(handle_unknown='ignore') # One-hot encoding for categorical features (ignores unseen labels during transform)

    # ---- Combine into a single ColumnTransformer ----
    preprocessor = ColumnTransformer( # Combines transformations for numeric and categorical data
        transformers=[
            ('num', numeric_transformer, numeric_features), # Apply numeric transformer to numeric columns
            ('cat', categorical_transformer, categorical_features) # Apply categorical transformer to categorical columns
        ]
    )

    return preprocessor # Return the preprocessing pipeline to be used in modeling
