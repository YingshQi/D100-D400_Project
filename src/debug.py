import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# Import pipelines if defined in another file
from model_pipelines import glm_pipeline, lgbm_pipeline  # Replace 'model_pipelines' with the correct file name if needed

# Define paths
project_root = Path().resolve().parent  # Adjust as needed
data_path = project_root / "data"

# Load the validation set
X_test = pd.read_parquet(data_path / "X_test_transformed.parquet")
y_test = pd.read_parquet(data_path / "y_test.parquet")

# Flatten y_test
y_test = y_test.values.ravel()

# Make predictions on the validation set
y_pred_glm_log = glm_pipeline.predict(X_test)
y_pred_lgbm_log = lgbm_pipeline.predict(X_test)

# Debug: Check the ranges and sample values of log-transformed target and predictions
print("--- Debug: Target and Predictions in Log Scale ---")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("y_test (log scale):", y_test[:5])
print("Range of y_test (log scale):", y_test.min(), y_test.max())
print("GLM Predictions (log scale):", y_pred_glm_log[:5])
print("Range of GLM Predictions (log scale):", y_pred_glm_log.min(), y_pred_glm_log.max())
print("LGBM Predictions (log scale):", y_pred_lgbm_log[:5])
print("Range of LGBM Predictions (log scale):", y_pred_lgbm_log.min(), y_pred_lgbm_log.max())

# Reverse log transformation if necessary
y_test_original = np.expm1(y_test)  # Reverse log transformation for the target variable
y_pred_glm = np.expm1(y_pred_glm_log)  # Reverse log transformation for GLM predictions
y_pred_lgbm = np.expm1(y_pred_lgbm_log)  # Reverse log transformation for LGBM predictions

# Debug: Check the ranges and sample values after log reversal
print("--- Debug: Log Reversal ---")
print("y_test_original:", y_test_original[:5])
print("Range of y_test_original:", y_test_original.min(), y_test_original.max())
print("GLM Predictions (Original Scale):", y_pred_glm[:5])
print("Range of GLM Predictions (Original Scale):", y_pred_glm.min(), y_pred_glm.max())
print("LGBM Predictions (Original Scale):", y_pred_lgbm[:5])
print("Range of LGBM Predictions (Original Scale):", y_pred_lgbm.min(), y_pred_lgbm.max())

# Evaluate GLM
glm_mse = mean_squared_error(y_test_original, y_pred_glm)
glm_mae = mean_absolute_error(y_test_original, y_pred_glm)
glm_r2 = r2_score(y_test_original, y_pred_glm)
print("--- Debug: GLM Evaluation ---")
print(f"- Mean Squared Error (MSE): {glm_mse}")
print(f"- Mean Absolute Error (MAE): {glm_mae}")
print(f"- R-squared (R2): {glm_r2}")

# Evaluate LGBM
lgbm_mse = mean_squared_error(y_test_original, y_pred_lgbm)
lgbm_mae = mean_absolute_error(y_test_original, y_pred_lgbm)
lgbm_r2 = r2_score(y_test_original, y_pred_lgbm)
print("--- Debug: LGBM Evaluation ---")
print(f"- Mean Squared Error (MSE): {lgbm_mse}")
print(f"- Mean Absolute Error (MAE): {lgbm_mae}")
print(f"- R-squared (R2): {lgbm_r2}")

# Debug: Residuals
print("--- Debug: Residuals ---")
glm_residuals = y_test_original - y_pred_glm
lgbm_residuals = y_test_original - y_pred_lgbm
print("GLM Residuals:", glm_residuals[:5])
print("LGBM Residuals:", lgbm_residuals[:5])

# Plot Residuals Distribution
plt.figure(figsize=(8, 6))
plt.hist(glm_residuals, bins=30, alpha=0.5, label='GLM Residuals', color='blue')
plt.hist(lgbm_residuals, bins=30, alpha=0.5, label='LGBM Residuals', color='orange')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Predicted vs Actual Plot for GLM
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_glm, alpha=0.6, label='GLM Predictions', color='blue')
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.title("Predicted vs Actual: GLM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

# Predicted vs Actual Plot for LGBM
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original, y_pred_lgbm, alpha=0.6, label='LGBM Predictions', color='green')
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         color='red', linestyle='--', label='Perfect Fit')
plt.title("Predicted vs Actual: LGBM")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()
