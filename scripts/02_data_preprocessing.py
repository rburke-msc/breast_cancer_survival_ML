import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

# ----------------------------------------------------
# Function: load_raw_data
# Description:
# Loads and preprocesses the raw breast cancer dataset.
# Handles tumor grade cleaning, feature engineering,
# and returns structured features and survival targets.
# ----------------------------------------------------
def load_raw_data(path="../data/raw/breast_cancer_data_raw.csv"):
    # Load dataset from CSV
    df = pd.read_csv(path)

    # ---- Clean and standardize Tumor Grade ----
    # Convert '1 or 2' into a numerical average (1.5)
    df["Tumor grade"] = df["Tumor grade"].replace('1 or 2', '1.5')
    df["Tumor grade"] = df["Tumor grade"].astype(float)  # Ensures numeric format

    # ---- Feature Engineering: Metastatic Sites ----
    # Map textual labels to binary numeric values
    df['N of met. Sites numeric'] = df['N of met. Sites'].map({'< 3': 0, '>=3': 1})


    # ---- Define categorical variables for encoding ----
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade']

    # Drop rows with missing categorical values
    df = df.dropna(subset=categorical_features)

    # ---- Define the model features and target ----
    feature_cols = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline'] + categorical_features
    features = df[feature_cols]

    # Survival target needs to be structured for survival analysis
    target = Surv.from_dataframe("Status OS", "OS at baseline", df)

    return features, target


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
    numeric_features = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline']
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade']

    # ---- Numeric Transformer ----
    # StandardScaler will normalize values to mean 0 and variance 1
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # ---- Categorical Transformer ----
    # OneHotEncoder will create binary columns for each category
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # ---- Combine into a single ColumnTransformer ----
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
