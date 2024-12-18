from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from pathlib import Path

# Define paths
project_root = Path().resolve().parent  
data_path = project_root / "data"

# Load transformed data
X_train = pd.read_parquet(data_path / "X_train_transformed.parquet")
y_train = pd.read_parquet(data_path / "y_train.parquet")
y_train = y_train.values.ravel()  # Flatten target array


# General preprocessor for missing values
preprocessor = SimpleImputer(strategy='mean')

# GLM (ElasticNet) Pipeline
glm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet())
])

# Define parameter grid for GLM
glm_param_grid = {
    'regressor__alpha': [0.1, 1.0, 10.0],  # Regularization strength
    'regressor__l1_ratio': [0.2, 0.5, 0.8]  # Elastic net mixing
}

# LGBM Pipeline
lgbm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor())
])

# Define parameter grid for LGBM
lgbm_param_grid = {
    'regressor__n_estimators': [100, 500, 1000],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5, 10]
}

# GridSearchCV for GLM
glm_search = GridSearchCV(
    estimator=glm_pipeline,
    param_grid=glm_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit GLM model with hyperparameter tuning
print("Tuning GLM...")
glm_search.fit(X_train, y_train)

print(f"Best GLM Params: {glm_search.best_params_}")
print(f"Best GLM MSE: {-glm_search.best_score_}")

# GridSearchCV for LGBM
lgbm_search = GridSearchCV(
    estimator=lgbm_pipeline,
    param_grid=lgbm_param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# Fit LGBM model with hyperparameter tuning
print("Tuning LGBM...")
lgbm_search.fit(X_train, y_train)

print(f"Best LGBM Params: {lgbm_search.best_params_}")
print(f"Best LGBM MSE: {-lgbm_search.best_score_}")
