import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import sys
from sklearn.impute import SimpleImputer
from pathlib import Path


# Simplified Custom StandardScaler
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.means_ = X.mean(axis=0)
        self.stds_ = X.std(axis=0, ddof=0)  # Population standard deviation
        self.stds_.replace(0, 1, inplace=True)  # Avoid division by zero
        return self

    def transform(self, X):
        return (X - self.means_) / self.stds_






# Custom OneHotEncoder
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X)
        self.columns = self.encoder.get_feature_names_out(X.columns)
        return self

    def transform(self, X):
        encoded_data = self.encoder.transform(X)
        return pd.DataFrame(encoded_data, columns=self.columns).fillna(0)




# Feature Engineering Pipeline
class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.custom_scaler = CustomStandardScaler()
        self.custom_encoder = CustomOneHotEncoder()
        self.numerical_imputer = SimpleImputer(strategy="mean")  # Handle missing values in numerical features
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")  # Handle missing values in categorical features

    def fit(self, X, y=None):
        # Fit imputers and transformers
        self.numerical_imputer.fit(X[self.numerical_features])  # Fit on numerical features
        self.custom_scaler.fit(X[self.numerical_features])  # Fit scaler on numerical features
        self.custom_encoder.fit(X[self.categorical_features])  # Fit encoder on categorical features
        self.categorical_imputer.fit(X[self.categorical_features])  # Fit imputer on categorical features
        return self

    def transform(self, X):
        # Handle missing values
        X_num = pd.DataFrame(self.numerical_imputer.transform(X[self.numerical_features]), 
                             columns=self.numerical_features)
        
        # Apply transformations
        X_num_scaled = pd.DataFrame(self.custom_scaler.transform(X_num), columns=self.numerical_features)
        
        # Handle missing values in categorical features
        X_cat = pd.DataFrame(self.categorical_imputer.transform(X[self.categorical_features]), 
                             columns=self.categorical_features)
        
        # Apply one-hot encoding
        X_cat_encoded = self.custom_encoder.transform(X_cat)
        
        return pd.concat([X_num_scaled, X_cat_encoded], axis=1)



# Load Split Data
def load_split_data(data_path):
    X_train = pd.read_parquet(f"{data_path}/X_train.parquet")
    X_test = pd.read_parquet(f"{data_path}/X_test.parquet")
    y_train = pd.read_parquet(f"{data_path}/y_train.parquet")
    y_test = pd.read_parquet(f"{data_path}/y_test.parquet")
    return X_train, X_test, y_train, y_test


# Main Script
def main():

    project_root = Path().resolve().parent  
    sys.path.append(str(project_root))
    data_path = project_root / "data"
    

    # Load split data
    X_train, X_test, y_train, y_test = load_split_data(data_path)

    # Define features
    numerical_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # Setup feature engineering pipeline
    fe_pipeline = FeatureEngineeringPipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    # Fit pipeline on training data and transform
    fe_pipeline.fit(X_train)
    X_train_transformed = fe_pipeline.transform(X_train)
    X_test_transformed = fe_pipeline.transform(X_test)

    # Save transformed data
    X_train_transformed.to_parquet(f"{data_path}/X_train_transformed.parquet")
    X_test_transformed.to_parquet(f"{data_path}/X_test_transformed.parquet")

    print("Feature engineering completed and transformed data saved!")


if __name__ == "__main__":
    main()
