from sklearn.linear_model import LinearRegression  # Use LinearRegression for regression tasks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from pathlib import Path

# Define paths
project_root = Path().resolve().parent  # Adjust as needed
data_path = project_root / "data"

# Load the transformed data
X_train = pd.read_parquet(data_path / "X_train_transformed.parquet")
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_train = pd.read_parquet(data_path / "y_train.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_train and y_test
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Apply log transformation to the target variable to reduce skewness
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# General preprocessor (imputation for numerical features)
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Preprocessor for handling missing values using SimpleImputer
preprocessor = SimpleImputer(strategy='mean')  # Impute missing values with the mean

# GLM Model (Linear Regression) Pipeline

glm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.2))  # Use best alpha and l1_ratio
])

"""
# before tuning
glm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  # Use LinearRegression for regression tasks
])
"""

# Fit and evaluate GLM model
glm_pipeline.fit(X_train, y_train_log)
y_pred_glm = glm_pipeline.predict(X_test)
mse_glm = mean_squared_error(y_test_log, y_pred_glm)
print(f"GLM Mean Squared Error: {mse_glm}")


"""
# LGBM Model (Gradient Boosting) Pipeline
# before tuning
lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])
"""

lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        n_estimators=1000
    ))
])

# Fit and evaluate LGBM model
lgbm_pipeline.fit(X_train, y_train_log)
y_pred_lgbm = lgbm_pipeline.predict(X_test)
mse_lgbm = mean_squared_error(y_test_log, y_pred_lgbm)
print(f"LGBM Mean Squared Error: {mse_lgbm}")

# Optional: Use cross-validation for model evaluation
cv_glm = cross_val_score(glm_pipeline, X_train, y_train_log, cv=5, scoring='neg_mean_squared_error')
cv_lgbm = cross_val_score(lgbm_pipeline, X_train, y_train_log, cv=5, scoring='neg_mean_squared_error')

print(f"GLM Cross-Validation MSE: {-cv_glm.mean()}")
print(f"LGBM Cross-Validation MSE: {-cv_lgbm.mean()}")
