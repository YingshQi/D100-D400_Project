from sklearn.linear_model import LinearRegression  
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths
project_root = Path().resolve().parent  
data_path = project_root / "data"

# Load the transformed data
X_train = pd.read_parquet(data_path / "X_train_transformed.parquet")
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_train = pd.read_parquet(data_path / "y_train.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_train and y_test
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# General preprocessor (imputation for numerical features)
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessor for handling missing values using SimpleImputer
preprocessor = SimpleImputer(strategy='mean')  # Impute missing values with the mean

# GLM Model (ElasticNet) Pipeline
# after tuning
glm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.2))  # Use best alpha and l1_ratio
])

# Fit and evaluate GLM model
glm_pipeline.fit(X_train, y_train)  # Use y_train directly (already log-transformed)
y_pred_glm = glm_pipeline.predict(X_test)
mse_glm = mean_squared_error(y_test, y_pred_glm)  # Compare with y_test directly
print(f"GLM Mean Squared Error: {mse_glm}")


# LGBM Model (Gradient Boosting) Pipeline
# after tuning
lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        learning_rate=0.01,
        max_depth=5,
        min_samples_split=2,
        n_estimators=1000
    ))
])

# Fit and evaluate LGBM model
lgbm_pipeline.fit(X_train, y_train)  # Use y_train directly (already log-transformed)
y_pred_lgbm = lgbm_pipeline.predict(X_test)
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)  # Compare with y_test directly
print(f"LGBM Mean Squared Error: {mse_lgbm}")

# Use cross-validation for model evaluation
cv_glm = cross_val_score(glm_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_lgbm = cross_val_score(lgbm_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

print(f"GLM Cross-Validation MSE: {-cv_glm.mean()}")
print(f"LGBM Cross-Validation MSE: {-cv_lgbm.mean()}")
