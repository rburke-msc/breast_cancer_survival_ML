import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.util import Surv

def load_raw_data(path="../data/raw/breast_cancer_data_raw.csv"):
    df = pd.read_csv(path)

    # Clean Tumor Grade
    df["Tumor grade"] = df["Tumor grade"].replace('1 or 2', '1.5')
    df["Tumor grade"] = df["Tumor grade"].astype(float)  # Keep NaNs for now

    # Feature Engineering
    df['N of met. Sites numeric'] = df['N of met. Sites'].map({'< 3': 0, '>=3': 1})

    # Define categorical features for model
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade']

    # Drop rows with missing categorical values
    df = df.dropna(subset=categorical_features)

    # Select features and target
    feature_cols = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline'] + categorical_features
    features = df[feature_cols]
    target = Surv.from_dataframe("Status OS", "OS at baseline", df)

    return features, target

def get_preprocessor():
    numeric_features = ['AGE', 'PS', 'CTCs counts at baseline', 'MAF of gene used at baseline']
    categorical_features = ['subtype', 'metastatic site', 'N of met. Sites numeric', 'Tumor grade']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor
